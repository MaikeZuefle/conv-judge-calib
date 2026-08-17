import csv
import tempfile
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


def get_audio_duration(convo_id):
    import soundfile as sf

    return sf.info(get_audio_path(convo_id)).duration


def _transcript_turns(convo_id):
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

    turns = []
    for row in rows:
        turn = {"label": labels[row["speaker"].strip()], "utterance": row["utterance"], "start": float(row["start"])}
        if row["backchannel"]:
            turn["backchannel_label"] = labels[row["backchannel_speaker"].strip()]
            turn["backchannel_text"] = row["backchannel"]
        turns.append(turn)
    return turns


def _format_transcript(turns):
    lines = []
    for turn in turns:
        line = f"{turn['label']}: {turn['utterance']}"
        if "backchannel_text" in turn:
            line += f" [{turn['backchannel_label']} backchannel: {turn['backchannel_text']}]"
        lines.append(line)
    return "\n".join(lines)


def get_transcript(convo_id):
    return _format_transcript(_transcript_turns(convo_id))


def _part_boundaries(turns, n_parts):
    """Return the n_parts-1 cut times (in seconds) splitting turns into n_parts, each cut
    snapped to the start of the closest actual turn so a cut never lands mid-utterance."""
    total = turns[-1]["start"]
    targets = [total * i / n_parts for i in range(1, n_parts)]
    return [min(turns, key=lambda t: abs(t["start"] - target))["start"] for target in targets]


def _turns_for_part(turns, part_idx, n_parts):
    boundaries = _part_boundaries(turns, n_parts)
    lo = boundaries[part_idx - 1] if part_idx > 0 else 0.0
    hi = boundaries[part_idx] if part_idx < len(boundaries) else None
    return [t for t in turns if t["start"] >= lo and (hi is None or t["start"] < hi)]


def get_transcript_part(convo_id, part_idx, n_parts=3):
    turns = _turns_for_part(_transcript_turns(convo_id), part_idx, n_parts)
    return _format_transcript(turns)


def get_audio_part_path(convo_id, part_idx, n_parts=3):
    import soundfile as sf

    turns = _transcript_turns(convo_id)
    boundaries = _part_boundaries(turns, n_parts)
    lo = boundaries[part_idx - 1] if part_idx > 0 else 0.0
    hi = boundaries[part_idx] if part_idx < len(boundaries) else None

    data, samplerate = sf.read(get_audio_path(convo_id))
    start_sample = int(lo * samplerate)
    end_sample = int(hi * samplerate) if hi is not None else len(data)

    out_path = Path(tempfile.gettempdir()) / f"{convo_id}_part{part_idx}of{n_parts}.wav"
    sf.write(out_path, data[start_sample:end_sample], samplerate)
    return str(out_path)


def get_input_part(convo_id, modality, part_idx, n_parts=3):
    if modality == "text":
        return get_transcript_part(convo_id, part_idx, n_parts)
    return get_audio_part_path(convo_id, part_idx, n_parts)


def _format_timestamp(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def get_transcript_html(convo_id):
    speaker_tags = {"Speaker A": "b", "Speaker B": "strong"}
    lines = []
    for turn in _transcript_turns(convo_id):
        tag = speaker_tags.get(turn["label"], "b")
        timestamp = (
            f'<a class="ts" onclick="this.closest(\'.output_tgt\').querySelector(\'audio\').currentTime='
            f'{turn["start"]}">[{_format_timestamp(turn["start"])}]</a> '
        )
        line = f"{timestamp}<{tag}>{turn['label']}:</{tag}> {turn['utterance']}"
        if "backchannel_text" in turn:
            line += f" <i>[{turn['backchannel_label']} backchannel: {turn['backchannel_text']}]</i>"
        lines.append(f"<p>{line}</p>")
    return "".join(lines)


def get_input(convo_id, modality):
    return get_transcript(convo_id) if modality == "text" else get_audio_path(convo_id)


def get_label(convo_id):
    return float(_metrics()[convo_id]["paper_pcs_proxy"])


def get_survey_values(convo_id, column):
    path = hf_hub_download(REPO_ID, repo_type="dataset", filename=f"{convo_id}/survey.csv")
    with open(path) as f:
        return [float(row[column]) for row in csv.DictReader(f) if row[column]]


def get_category(convo_id):
    return _metrics()[convo_id]["label"]
