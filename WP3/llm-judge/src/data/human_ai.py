import csv
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from scipy.optimize import minimize

REPO_ID = "patuchen/Human-AI-interaction"
NATURAL_DIR = "Human-AI-Natural-Conversation-Recording"
ARENA_DIR = "Human-Agent-VoiceArena"
ARENA_TRANSCRIPTS_PATH = Path("outputs") / "voice_arena_asr" / "transcripts.jsonl"


@cache
def _root():
    return Path(snapshot_download(REPO_ID, repo_type="dataset", local_files_only=True))


# ---------------------------------------------------------------------------
# Sub-dataset 1: Human-AI Natural Conversation Recording
# 5 participants x 4 open-domain spoken conversations with an AI voice agent.
# ---------------------------------------------------------------------------


@cache
def _recording_index():
    path = _root() / NATURAL_DIR / "recording_index.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def list_natural_recordings():
    return [row["anonymized_recording_folder"] for row in _recording_index()]


def _recording_row(recording_folder):
    for row in _recording_index():
        if row["anonymized_recording_folder"] == recording_folder:
            return row
    raise KeyError(recording_folder)


def get_natural_audio_path(recording_folder, speaker):
    """speaker: "human" or "ai". Returns the .wav path (a matching .mp3 sits alongside it)."""
    row = _recording_row(recording_folder)
    subfolder = row["ai_media_folder"] if speaker == "ai" else row["human_media_folder"]
    folder = _root() / NATURAL_DIR / recording_folder / subfolder
    wav_files = list(folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"no audio in {folder}")
    return str(wav_files[0])


def get_natural_transcript(recording_folder):
    row = _recording_row(recording_folder)
    path = _root() / NATURAL_DIR / recording_folder / row["transcription_file"]
    return path.read_text()


@cache
def _feedback_rows():
    path = _root() / NATURAL_DIR / "Natural Conversation Feedback.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_natural_feedback(recording_folder):
    """This recording's own feedback fields (naturalness rating, emotions, etc.), pulled out of
    the participant-level wide CSV (one row per participant, with columns "N.1".."N.6" for each
    of their 4 recordings, N=session order). Columns "N. ..." (no second number, e.g. "1. What
    were the most natural moments...") are separate, participant-level reflection questions not
    tied to any single recording, and are excluded here."""
    row = _recording_row(recording_folder)
    participant_id, recording_id = row["participant_id"], row["recording_id"]

    for feedback_row in _feedback_rows():
        if feedback_row["participant_id"] != participant_id:
            continue
        for i in range(1, 5):
            if feedback_row.get(f"recording_{i}") == recording_id:
                prefix = f"{i}."
                return {
                    k[len(prefix) :]: v
                    for k, v in feedback_row.items()
                    if k.startswith(prefix) and k[len(prefix) : len(prefix) + 1].isdigit()
                }
    return None


# ---------------------------------------------------------------------------
# Sub-dataset 2: Human-Agent VoiceArena
# Callers completing airline customer-service tasks against 5 agent systems.
# ---------------------------------------------------------------------------


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
