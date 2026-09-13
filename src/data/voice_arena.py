import csv
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from scipy.optimize import minimize

REPO_ID = "patuchen/Human-AI-interaction"
ARENA_DIR = "Human-Agent-VoiceArena"
ARENA_TRANSCRIPTS_PATH = Path("outputs") / "voice_arena_asr" / "transcripts.jsonl"


@cache
def _root():
    return Path(snapshot_download(REPO_ID, repo_type="dataset", local_files_only=True))


@cache
def _arena_metadata():
    path = _root() / ARENA_DIR / "metadata.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def list_calls():
    return [row["call_id"] for row in _arena_metadata()]


def get_call_metadata(call_id):
    for row in _arena_metadata():
        if row["call_id"] == call_id:
            return row
    raise KeyError(call_id)


def get_call_audio_path(call_id, channel="merged"):
    """channel: "merged", "user", or "agent"."""
    return str(_root() / ARENA_DIR / call_id / f"{channel}.wav")


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
    path = _root() / ARENA_DIR / call_id / "tool_log.json"
    with open(path) as f:
        return json.load(f)


@cache
def _arena_transcripts():
    if not ARENA_TRANSCRIPTS_PATH.exists():
        return {}
    with open(ARENA_TRANSCRIPTS_PATH) as f:
        return {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}


def get_call_transcript(call_id):
    """Turn-by-turn transcript produced by transcribe_voice_arena.py (VoiceArena ships no
    transcript of its own), interleaving user/agent segments chronologically by ASR timestamp.
    Returns None if that call hasn't been transcribed yet."""
    row = _arena_transcripts().get(call_id)
    if row is None:
        return None
    return "\n".join(f"{segment['speaker']}: {segment['text']}" for segment in row["segments"])


@cache
def get_votes():
    """Pairwise human-preference votes comparing two calls' agent providers."""
    path = _root() / ARENA_DIR / "vote_log.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_call_id(raw):
    # vote_log.csv stores call ids as floats (e.g. "417.0"); metadata uses plain "417"
    return str(int(float(raw)))


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
    average-strength opponent), fit from vote_log.csv's sparse pairwise comparisons (only ~2%
    of all possible call pairs are actually compared, so a raw win rate would be biased by which
    opponents each call happened to draw). Ties count as half a win for each side.
    criterion: "task_capability" or "humanness"."""
    winner_col = f"{criterion}_winner"
    call_ids = list_calls()
    comparisons = []
    for v in get_votes():
        c1, c2 = normalize_call_id(v["call_id_1"]), normalize_call_id(v["call_id_2"])
        winner = v[winner_col]
        if winner == "Tie":
            comparisons.append((c1, c2, 0.5))
            comparisons.append((c2, c1, 0.5))
        elif winner == v["model1"]:
            comparisons.append((c1, c2, 1.0))
        elif winner == v["model2"]:
            comparisons.append((c2, c1, 1.0))
    return _bradley_terry_strengths(call_ids, comparisons)


def list_conversations():
    return list_calls()


def get_input(convo_id, modality):
    if modality == "text":
        transcript = get_call_transcript(convo_id)
        if transcript is None:
            raise FileNotFoundError(f"no transcript for call {convo_id} yet: run transcribe_voice_arena.py first")
        return transcript
    return get_call_audio_mono_path(convo_id)


def get_label(convo_id):
    """Bradley-Terry strength (0-1) on whether the call accomplished the airline task, the most
    direct analog to "success" for this task-oriented dataset."""
    return get_call_strengths("task_capability")[convo_id]


def get_category(convo_id):
    return get_call_metadata(convo_id)["provider"]
