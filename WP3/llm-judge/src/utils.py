import json
import re
from pathlib import Path

LEVEL_COLORS = {"INFO": "92", "WARN": "93", "ERROR": "91"}


def colorize(text: str, color: str) -> str:
    return f"\033[{color}m{text}\033[0m"


def log(level: str, msg: str):
    print(f"{colorize(f'[{level}]', LEVEL_COLORS.get(level, '97'))} {msg}")


def get_run_dir(args):
    return Path(args.output_folder) / f"{args.dataset}_{args.model}_{args.prompt}_{args.input_modality}"


def get_examples_path(args):
    return get_run_dir(args) / "examples.jsonl"


def load_done_ids(output_path):
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def filter_convo_ids(dataset, convo_ids, categories, limit):
    if categories is not None:
        keep = set(categories.split(","))
        convo_ids = [convo_id for convo_id in convo_ids if dataset.get_category(convo_id) in keep]
        log("INFO", f"filtered to categories {sorted(keep)}: {len(convo_ids)} conversations remain")
    if limit is not None:
        convo_ids = convo_ids[:limit]
        log("INFO", f"limited to first {len(convo_ids)} conversations")
    return convo_ids


def log_resume_status(done_ids, total):
    if done_ids:
        log("INFO", f"resuming: {len(done_ids)}/{total} examples already done")
    else:
        log("INFO", f"starting: 0/{total} examples done")


def extract_json_block(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else None


def parse_json_dict(output):
    for candidate in (output, extract_json_block(output)):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
