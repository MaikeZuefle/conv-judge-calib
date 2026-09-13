import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from calibration_functions import METHODS, loss_wasserstein

CALIBRATION_DIR = Path(__file__).resolve().parent

DEFAULT_JUDGES = [
    "phi4mm_speech",
    "phi4mm_text",
    "qwen3omni_speech",
    "qwen3omni_text",
    "qwen35_text",
    "qwen25omni_success_speech",
    "qwen25omni_success_text",
]


def load_data(judges, judge_scores_file):
    data_llm_raw = [json.loads(line) for line in open(judge_scores_file)]
    data_human_raw = [json.loads(line) for line in open(CALIBRATION_DIR / "pointwise_dataset.jsonl")]

    data_agg = {j: {} for j in judges}
    for row in data_llm_raw:
        if row["judge"] in data_agg and row["score"] is not None:
            data_agg[row["judge"]][row["conversation"]] = min(10, max(0, row["score"]))

    data_human = {row["conversation"]: row["score"] * 10
                  for row in data_human_raw if row["conversation"].startswith("CANDOR") and not row["mock"]}

    common = sorted(set.intersection(set(data_human), *[set(s) for s in data_agg.values()]))
    human = [data_human[c] for c in common]
    agg = {j: [data_agg[j][c] for c in common] for j in judges}
    return common, human, agg


def random_anchor(n, size, rng):
    return rng.choice(n, size=size, replace=False).tolist()


def optimized_anchor(human, other_judges_scores, size, transform, rng, n_candidates=1000):
    best_indices, best_loss = None, float("inf")
    for _ in range(n_candidates):
        indices = random_anchor(len(human), size, rng)
        anchor_human = [human[i] for i in indices]
        losses = []
        for scores in other_judges_scores:
            anchor_scores = [scores[i] for i in indices]
            transformed = np.clip(transform(anchor_human, anchor_scores, scores), 0, 10).tolist()
            losses.append(loss_wasserstein(human, transformed))
        loss = statistics.mean(losses)
        if loss < best_loss:
            best_loss, best_indices = loss, indices
    return best_indices


def score_anchor(human, scores, indices, transform):
    anchor_human = [human[i] for i in indices]
    anchor_scores = [scores[i] for i in indices]
    eval_idx = [i for i in range(len(scores)) if i not in set(indices)]
    eval_human = [human[i] for i in eval_idx]
    eval_scores = [scores[i] for i in eval_idx]
    transformed = np.clip(transform(anchor_human, anchor_scores, eval_scores), 0, 10).tolist()
    return loss_wasserstein(eval_human, transformed)


def main(args):
    rng = np.random.default_rng(args.seed)
    common, human, agg = load_data(args.judges, args.judge_scores_file)
    print(f"n_conversations={len(common)} judges={args.judges}\n")

    header = f"{'Method':<20} {'rand-8':>7} {'rand-32':>7} {'rand-128':>8} {'opt-8':>7} {'opt-32':>8} {'opt-128':>8}"
    print(header)
    for name, transform in METHODS:
        row = [name]
        for size in (8, 32, 128):
            random_losses = []
            for _ in range(10):
                indices = random_anchor(len(human), size, rng)
                for scores in agg.values():
                    random_losses.append(score_anchor(human, scores, indices, transform))
            row.append(statistics.mean(random_losses))
        for size in (8, 32, 128):
            optimized_losses = []
            for judge, scores in agg.items():
                others = [s for j, s in agg.items() if j != judge]
                indices = optimized_anchor(human, others, size, transform, rng)
                optimized_losses.append(score_anchor(human, scores, indices, transform))
            row.append(statistics.mean(optimized_losses))
        print(f"{row[0]:<20} {row[1]:>7.3f} {row[2]:>7.3f} {row[3]:>8.3f} {row[4]:>7.3f} {row[5]:>8.3f} {row[6]:>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--judge_scores_file", default=str(CALIBRATION_DIR / "judge_scores_dataset.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
