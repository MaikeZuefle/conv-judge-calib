import argparse
import json

from tqdm import tqdm
from transformers import set_seed

from data import candor, voice_arena
from prompts import PROMPTS
from models.qwen25omni import Qwen25Omni
from models.avflamingo import AVFlamingo
from models.phi4multimodal import Phi4Multimodal
from models.qwen36_27b import Qwen36_27B
from utils import filter_convo_ids, get_examples_path, load_done_ids, log, log_resume_status

DATASETS = {"candor": candor, "voice_arena": voice_arena}
MODELS = {
    "qwen25omni": Qwen25Omni,
    "avflamingo": AVFlamingo,
    "phi4multimodal": Phi4Multimodal,
    "qwen36_27b": Qwen36_27B,
}


def main(args):
    set_seed(args.seed)

    dataset = DATASETS[args.dataset]
    output_path = get_examples_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_path)

    log("INFO", f"listing conversations for {args.dataset}")
    convo_ids = filter_convo_ids(dataset, dataset.list_conversations(), args.categories, args.limit)
    prompt = PROMPTS[args.prompt]

    log("INFO", f"loading model {args.model}")
    model = MODELS[args.model]()

    log_resume_status(done_ids, len(convo_ids))

    pending = (convo_id for convo_id in convo_ids if convo_id not in done_ids)
    with open(output_path, "a") as f:
        for convo_id in tqdm(pending, initial=len(done_ids), total=len(convo_ids)):
            content = dataset.get_input(convo_id, args.input_modality)
            output = model.generate(prompt, content, args.input_modality)
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
    parser.add_argument("--input_modality", default="speech", choices=["speech", "text"])
    parser.add_argument("--output_folder", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default=None, help="comma-separated category allowlist, e.g. HSC,LSC")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N conversations")
    args = parser.parse_args()
    main(args)
