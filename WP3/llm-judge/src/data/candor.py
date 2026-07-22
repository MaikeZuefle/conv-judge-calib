import csv
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download
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
def _snapshot_dir():
    patterns = [f"{convo_id}/processed/{convo_id}.mp3" for convo_id in _metrics()]
    return Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=patterns))


def list_conversations():
    _snapshot_dir()
    return sorted(_metrics())


def get_audio_path(convo_id):
    return str(_snapshot_dir() / convo_id / "processed" / f"{convo_id}.mp3")


def get_label(convo_id):
    return float(_metrics()[convo_id]["paper_pcs_proxy"])


def get_category(convo_id):
    return _metrics()[convo_id]["label"]
