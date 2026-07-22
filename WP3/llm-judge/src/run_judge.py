import argparse
import json

from tqdm import tqdm
from transformers import set_seed

from data import candor
from prompts import PROMPTS
from models.qwen25omni import Qwen25Omni
from utils import get_output_path, log

DATASETS = {"candor": candor}
MODELS = {"qwen25omni": Qwen25Omni}


def load_done_ids(output_path):
    if not output_path.exists():
        return set()
    with open(output_path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main(args):
    set_seed(args.seed)

    dataset = DATASETS[args.dataset]
    output_path = get_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    log("INFO", f"listing conversations for {args.dataset}")
    convo_ids = dataset.list_conversations()
    prompt = PROMPTS[args.prompt]

    log("INFO", f"loading model {args.model}")
    model = MODELS[args.model]()

    if done_ids:
        log("INFO", f"resuming: {len(done_ids)}/{len(convo_ids)} examples already done")
    else:
        log("INFO", f"starting: 0/{len(convo_ids)} examples done")

    pending = (convo_id for convo_id in convo_ids if convo_id not in done_ids)
    with open(output_path, "a") as f:
        for convo_id in tqdm(pending, initial=len(done_ids), total=len(convo_ids)):
            audio = dataset.get_audio_path(convo_id)
            output = model.generate(prompt, audio)
            label = dataset.get_label(convo_id)
            category = dataset.get_category(convo_id)
            f.write(json.dumps({"id": convo_id, "output": output, "label": label, "category": category}) + "\n")
            f.flush()

    log("INFO", "finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="candor", choices=list(DATASETS))
    parser.add_argument("--model", default="qwen25omni", choices=list(MODELS))
    parser.add_argument("--prompt", default="success", choices=list(PROMPTS))
    parser.add_argument("--output_folder", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
