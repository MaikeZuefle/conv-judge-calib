"""Transcribe every VoiceArena call once with Whisper, into the file voice_arena.get_call_transcript
reads (the dataset ships no transcripts). Resumable: already transcribed calls are skipped.

Each call has separate user and agent channels, so speaker attribution is exact rather than
diarised: the two channels are transcribed independently and merged by timestamp.
"""

import argparse
import json

from tqdm import tqdm

from data import voice_arena
from utils import filter_convo_ids, load_done_ids, log, log_resume_status

# the labels the VoiceArena prompts refer to ("the Caller, or the Agent")
SPEAKERS = {"user": "Caller", "agent": "Agent"}


def transcribe_channel(model, path):
    """Whisper segments for one channel, as [(start, end, text)]."""
    import librosa

    audio, _ = librosa.load(path, sr=16000, mono=True)
    result = model.transcribe(audio, language="en")
    return [(float(s["start"]), float(s["end"]), s["text"].strip()) for s in result["segments"] if s["text"].strip()]


def transcribe_call(model, call_id):
    segments = []
    for channel, speaker in SPEAKERS.items():
        for start, end, text in transcribe_channel(model, voice_arena.get_call_audio_path(call_id, channel)):
            segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return sorted(segments, key=lambda segment: segment["start"])


def main(args):
    output_path = voice_arena.ARENA_TRANSCRIPTS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    call_ids = filter_convo_ids(voice_arena, voice_arena.list_calls(), None, args.limit)

    import whisper

    log("INFO", f"loading whisper {args.model}")
    model = whisper.load_model(args.model)

    log_resume_status(done_ids, len(call_ids))

    pending = (call_id for call_id in call_ids if call_id not in done_ids)
    with open(output_path, "a") as f:
        for call_id in tqdm(pending, initial=len(done_ids), total=len(call_ids)):
            f.write(json.dumps({"id": call_id, "segments": transcribe_call(model, call_id)}) + "\n")
            f.flush()

    log("INFO", "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small.en", help="whisper model; small.en matches earlier runs")
    parser.add_argument("--limit", type=int, default=None, help="only transcribe the first N calls")
    args = parser.parse_args()
    main(args)
