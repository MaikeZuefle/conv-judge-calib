import csv
from functools import cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils.logging import set_verbosity_error

disable_progress_bars()
set_verbosity_error()

REPO_ID = "JSALT2026-Conv-AI-Simulator/CANDOR"
METRICS_PATH = Path(__file__).parent / "survey_metrics_conversation_level.csv"


@cache
def _metrics():
    with open(METRICS_PATH) as f:
        return {row["convo_id"]: row for row in csv.DictReader(f)}


@cache
def _included_ids():
    return sorted(cid for cid, row in _metrics().items() if row["label"] != "UNSCORED" and row["paper_pcs_proxy"])


def list_conversations():
    return _included_ids()


def get_audio_path(convo_id):
    return hf_hub_download(REPO_ID, repo_type="dataset", filename=f"{convo_id}/processed/{convo_id}.mp3")


def get_transcript(convo_id):
    path = hf_hub_download(
        REPO_ID, repo_type="dataset", filename=f"{convo_id}/transcription/transcript_backbiter.csv"
    )
    with open(path) as f:
        rows = list(csv.DictReader(f))

    speaker_ids = []
    for row in rows:
        speaker_id = row["speaker"].strip()
        if speaker_id not in speaker_ids:
            speaker_ids.append(speaker_id)
    labels = {speaker_id: f"Speaker {chr(65 + i)}" for i, speaker_id in enumerate(speaker_ids)}

    lines = []
    for row in rows:
        line = f"{labels[row['speaker'].strip()]}: {row['utterance']}"
        if row["backchannel"]:
            backchannel_label = labels[row["backchannel_speaker"].strip()]
            line += f" [{backchannel_label} backchannel: {row['backchannel']}]"
        lines.append(line)
    return "\n".join(lines)


def get_input(convo_id, modality):
    return get_transcript(convo_id) if modality == "text" else get_audio_path(convo_id)


def get_label(convo_id):
    return float(_metrics()[convo_id]["paper_pcs_proxy"])


def get_category(convo_id):
    return _metrics()[convo_id]["label"]
