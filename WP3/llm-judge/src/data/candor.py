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


def get_label(convo_id):
    return float(_metrics()[convo_id]["paper_pcs_proxy"])


def get_category(convo_id):
    return _metrics()[convo_id]["label"]
