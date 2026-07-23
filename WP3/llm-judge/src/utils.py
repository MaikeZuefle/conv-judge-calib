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
