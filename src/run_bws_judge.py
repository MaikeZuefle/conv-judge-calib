"""Run an LLM judge over best-worst-scaling batches and record its rankings.

Reads batches produced by build_bws_batches.py, prompts the model to rank each batch
from best to worst, and writes one result line per batch/ordering. Resumable: rerunning
skips (batch_id, ordering_idx) pairs already present in the output file.
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from transformers import set_seed

from data import voice_arena
from data.survey_features import load_transcript
from models.registry import ALL as MODELS
from prompts import (
    BWS_LAYOUTS,
    VOICEARENA_DIMENSIONS,
    bws_layout,
    feature_layout,
    voicearena_layout,
)
from utils import log, parse_json_dict

def load_done_keys(output_path):
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        return {(json.loads(line)["batch_id"], json.loads(line)["ordering_idx"])
                for line in f if line.strip()}


def parse_ranking(output, n, style="rank"):
    """Extract the judgement from the model output, or None if malformed.

    For the "best_only" style the model emits a bare number, so the result is a
    one-element list holding just the chosen best conversation; downstream scoring
    treats a short ranking as best-of-batch only."""
    if style == "prometheus":
        tail = output.rsplit("[RESULT]", 1)[-1] if "[RESULT]" in output else output
        match = re.search(r"\b([AB])\b", tail)
        return [{"A": 1, "B": 2}[match.group(1)]] if match else None

    if style == "best_only":
        # Reasoning output contains numerals throughout, so prefer an explicit
        # "ANSWER: k" marker and fall back to the first number only for terse replies.
        marked = re.findall(r"ANSWER:\s*(\d+)", output, re.IGNORECASE)
        match = marked[-1] if marked else None
        if match is None:
            plain = re.search(r"\d+", output)
            match = plain.group() if plain else None
        if match is None:
            return None
        best = int(match)
        return [best] if 1 <= best <= n else None

    data = parse_json_dict(output)
    if not isinstance(data, dict):
        return None
    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return None
    try:
        ranking = [int(v) for v in ranking]
    except (TypeError, ValueError):
        return None
    return ranking if sorted(ranking) == list(range(1, n + 1)) else None


def tail_tokens(text, tokenizer, max_tokens):
    """Keep the last `max_tokens` tokens of a transcript. Conversation endings carry more
    signal about how it went than the opening pleasantries do, so truncation keeps the
    tail. A partial first line is dropped so the excerpt starts on a speaker turn."""
    ids = tokenizer(text)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    excerpt = tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)
    return excerpt.split("\n", 1)[1] if "\n" in excerpt else excerpt


def main(args):
    set_seed(args.seed)

    batches = [json.loads(line) for line in open(args.batches_file) if line.strip()]
    if args.limit:
        batches = batches[:args.limit]
    log("INFO", f"{len(batches)} batch orderings to judge from {args.batches_file}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(output_path)
    log("INFO", f"resuming: {len(done)} already done")

    if args.transcript_source == "voicearena_audio":
        read_input = voice_arena.get_call_audio_mono_path
        input_modality = "speech"
    elif args.transcript_source == "voicearena":
        read_input = voice_arena.get_call_transcript
        input_modality = "text"
    else:
        read_input = load_transcript
        input_modality = "text"

    log("INFO", f"loading model {args.model}")
    model = MODELS[args.model]()
    tokenizer = getattr(model, "tokenizer", None) or model.processor.tokenizer

    pending = [b for b in batches if (b["batch_id"], b["ordering_idx"]) not in done]
    n_over_window, n_failed_parse, token_counts = 0, 0, []
    with open(output_path, "a") as f:
        for batch in tqdm(pending, initial=len(done), total=len(batches)):
            inputs = [read_input(cid) for cid in batch["convo_ids"]]
            if input_modality == "text" and args.truncate_tokens:
                inputs = [tail_tokens(t, tokenizer, args.truncate_tokens) for t in inputs]
            if args.voicearena_dimension:
                preamble, prompt, system = voicearena_layout(
                    len(inputs), args.voicearena_dimension, modality=input_modality,
                    cot=args.voicearena_cot)
                if input_modality == "speech":
                    content = inputs  # audio paths — model labels clips as [CALL i]
                else:
                    content = "\n\n".join(
                        f"[CALL {i}]\n{t}" for i, t in enumerate(inputs, 1))
            elif args.feature_question:
                preamble, prompt, system = feature_layout(
                    len(inputs), args.feature_question, cot=args.feature_cot)
                content = "\n\n".join(
                    f"[CONVERSATION {i}]\n{t}" for i, t in enumerate(inputs, 1)
                )
            elif args.parse_as == "prometheus":
                preamble, prompt, system = None, "", None
                content = model.build_prompt(inputs)
            else:
                preamble, prompt, system = bws_layout(len(inputs), args.layout)
                content = "\n\n".join(
                    f"[CONVERSATION {i}]\n{t}" for i, t in enumerate(inputs, 1)
                )
            # count before generating; for audio, count text tokens in prompt only
            if input_modality == "text":
                full = "\n\n".join(p for p in (preamble, content, prompt) if p)
                n_tokens = len(tokenizer(full)["input_ids"])
            else:
                full = "\n\n".join(p for p in (preamble, prompt) if p)
                n_tokens = len(tokenizer(full)["input_ids"])  # audio tokens counted separately by model
            token_counts.append(n_tokens)
            over_window = n_tokens > args.context_window
            n_over_window += over_window

            output = model.generate(prompt, content, modality=input_modality,
                                    max_new_tokens=args.max_new_tokens,
                                    preamble=preamble, system=system,
                                    clip_prefix="CALL")
            ranking = parse_ranking(output, len(inputs), args.parse_as)
            n_failed_parse += ranking is None
            f.write(json.dumps({
                **{k: batch[k] for k in ("batch_id", "ordering_idx", "n_convs", "convo_ids",
                                         "gold_ranking", "scores", "min_adjacent_gap")},
                # carried through when present so the scorer can break results out by
                # dimension and check the human-agent identity shortcut
                **{k: batch[k] for k in ("dimension", "providers", "n_votes") if k in batch},
                "n_tokens": n_tokens,
                "over_context_window": over_window,
                "output": output,
                "ranking": ranking,
            }) + "\n")
            f.flush()

    if token_counts:
        ordered = sorted(token_counts)
        log("INFO", f"prompt tokens: min={ordered[0]} median={ordered[len(ordered)//2]} max={ordered[-1]}")
        log("INFO", f"over {args.context_window} tokens: {n_over_window}/{len(token_counts)} "
                    f"({100*n_over_window/len(token_counts):.1f}%)")
        log("INFO", f"unparseable rankings: {n_failed_parse}/{len(token_counts)}")
    log("INFO", "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="qwen25omni", choices=list(MODELS))
    parser.add_argument("--layout", default="closing_rank", choices=list(BWS_LAYOUTS))
    parser.add_argument("--feature_question", default=None,
                        help="ask about one survey question; overrides --layout")
    parser.add_argument("--feature_cot", action="store_true")
    parser.add_argument("--voicearena_dimension", default=None,
                        choices=list(VOICEARENA_DIMENSIONS),
                        help="judge VoiceArena calls on this dimension; overrides --layout")
    parser.add_argument("--voicearena_cot", action="store_true",
                        help="prepend a brief reasoning instruction to the VoiceArena closing "
                             "and append an ANSWER marker; requires --parse_as best_only")
    parser.add_argument("--transcript_source", default="candor",
                        choices=("candor", "voicearena", "voicearena_audio"),
                        help="where to read input from; voicearena_audio passes merged WAV "
                             "files to audio-capable models instead of transcripts")
    parser.add_argument("--parse_as", default="rank",
                        choices=["rank", "best_only", "prometheus"])
    parser.add_argument("--truncate_tokens", type=int, default=None,
                        help="keep only the last N tokens of each transcript")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--context_window", type=int, default=32768,
                        help="prompts above this are flagged; the model may still handle them")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
