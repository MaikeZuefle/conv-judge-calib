import csv
import io
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from scipy.optimize import minimize

REPO_ID = "VoiceArena/Goal-Dataset_en_in"
VOTES_FILE = "pairwise_majority_votes.csv"
AUDIO_COLUMNS = ("merged_audio", "user_audio", "agent_audio")
ARENA_TRANSCRIPTS_PATH = Path("outputs") / "voice_arena_asr" / "transcripts.jsonl"


@cache
def _root():
    return Path(snapshot_download(REPO_ID, repo_type="dataset"))


@cache
def _arena_metadata():
    """{call_id: row} for every call, read without the large audio columns. Each row also
    records which parquet file holds the call, so its audio can be read from that file alone."""
    metadata = {}
    for path in sorted((_root() / "data").glob("*.parquet")):
        columns = [name for name in pq.read_schema(path).names if name not in AUDIO_COLUMNS]
        for row in pq.read_table(path, columns=columns).to_pylist():
            row["call_id"] = str(row["call_id"])
            row["parquet_file"] = path
            metadata[row["call_id"]] = row
    return metadata


def list_calls():
    return list(_arena_metadata())


def get_call_metadata(call_id):
    return _arena_metadata()[call_id]


def get_call_audio_path(call_id, channel="merged"):
    """channel: "merged", "user", or "agent". The dataset embeds audio as FLAC bytes in its
    parquet files; they are decoded once to a WAV file, since the judge models take a path."""
    import soundfile as sf

    out_path = Path(tempfile.gettempdir()) / f"{call_id}_{channel}.wav"
    if out_path.exists():
        return str(out_path)

    column = f"{channel}_audio"
    table = pq.read_table(
        get_call_metadata(call_id)["parquet_file"], columns=[column], filters=[("call_id", "=", int(call_id))]
    )
    data, samplerate = sf.read(io.BytesIO(table.column(column)[0].as_py()["bytes"]))
    sf.write(out_path, data, samplerate)
    return str(out_path)


def get_call_audio_mono_path(call_id):
    """merged.wav is stereo (one channel per speaker); this downmixes it to mono, since the
    judge models' audio-loading code isn't confirmed to handle stereo input safely."""
    import soundfile as sf

    out_path = Path(tempfile.gettempdir()) / f"{call_id}_merged_mono.wav"
    if out_path.exists():
        return str(out_path)

    data, samplerate = sf.read(get_call_audio_path(call_id, "merged"))
    mono = data.mean(axis=1) if data.ndim > 1 else data
    sf.write(out_path, mono, samplerate)
    return str(out_path)


def get_tool_log(call_id):
    return json.loads(get_call_metadata(call_id)["tool_calls_json"])


@cache
def _arena_transcripts():
    if not ARENA_TRANSCRIPTS_PATH.exists():
        return {}
    with open(ARENA_TRANSCRIPTS_PATH) as f:
        return {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}


def get_call_transcript(call_id):
    """Turn-by-turn transcript produced by transcribe_whisper.py (VoiceArena ships no
    transcript of its own), interleaving user/agent segments chronologically by ASR timestamp.
    Returns None if that call hasn't been transcribed yet."""
    row = _arena_transcripts().get(call_id)
    if row is None:
        return None
    return "\n".join(f"{segment['speaker']}: {segment['text']}" for segment in row["segments"])


@cache
def get_votes():
    """Pairwise human-preference votes comparing two calls' agent providers, one row per
    compared pair with the number of raters who preferred each call or called a tie."""
    with open(_root() / VOTES_FILE, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _bradley_terry_strengths(items, comparisons):
    """comparisons: list of (winner, loser, weight) triples. Returns {item: strength in [0, 1]},
    the modeled probability of beating an average-strength opponent."""
    index = {item: i for i, item in enumerate(items)}

    def neg_log_likelihood(w):
        ll = 0.0
        for winner, loser, weight in comparisons:
            i, j = index[winner], index[loser]
            ll += weight * (w[i] - np.logaddexp(w[i], w[j]))
        return -ll

    result = minimize(neg_log_likelihood, np.zeros(len(items)), method="L-BFGS-B")
    strength = result.x - result.x.mean()
    return {item: float(1 / (1 + np.exp(-strength[i]))) for item, i in index.items()}


@cache
def get_call_strengths(criterion="task_capability"):
    """Bradley-Terry strength score per call_id (0-1: modeled probability of beating an
    average-strength opponent), fit from the sparse pairwise votes (only ~2% of all possible
    call pairs are actually compared, so a raw win rate would be biased by which opponents each
    call happened to draw). Every rater's vote counts once; ties count as half a win for each side.
    criterion: "task_capability" or "humanness"."""
    prefix = {"task_capability": "task", "humanness": "humanness"}[criterion]
    call_ids = list_calls()
    comparisons = []
    for v in get_votes():
        a, b = v["call_id_a"], v["call_id_b"]
        ties = int(v[f"{prefix}_votes_tie"])
        comparisons.append((a, b, int(v[f"{prefix}_votes_a"]) + 0.5 * ties))
        comparisons.append((b, a, int(v[f"{prefix}_votes_b"]) + 0.5 * ties))
    return _bradley_terry_strengths(call_ids, comparisons)


def list_conversations():
    return list_calls()


def get_input(convo_id, modality):
    if modality == "text":
        transcript = get_call_transcript(convo_id)
        if transcript is None:
            raise FileNotFoundError(f"no transcript for call {convo_id} yet: run transcribe_whisper.py first")
        return transcript
    return get_call_audio_mono_path(convo_id)


def get_label(convo_id):
    """Bradley-Terry strength (0-1) on whether the call accomplished the airline task, the most
    direct analog to "success" for this task-oriented dataset."""
    return get_call_strengths("task_capability")[convo_id]


def get_category(convo_id):
    return get_call_metadata(convo_id)["provider"]
