"""Reproduces tab:calibration (WP3/vilem/scripts/02-plot_distribution.py), plus a held-out
variant: the original evaluates the transformed judge distribution on every conversation the
judge scored, anchor points included; --held_out restricts evaluation to conversations outside
the anchor, so the reported distance reflects generalization rather than the anchor fit itself.

Anchor selection: random draws `size` conversations uniformly; optimized searches 1000 random
candidates and keeps whichever anchor minimizes the mean Wasserstein loss for the OTHER judges
(leave-one-out), then fits and evaluates on the held-out judge -- so the anchor is chosen
without looking at the judge it is later scored on.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.stats

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration_data"

ORIGINAL_JUDGES = [
    "qwen25omni_success_text",
    "phi4multimodal_CoT_summary_text",
    "phi4multimodal_CoT_questions_summary_liking_text",
    "phi4multimodal_CoT_questions_speech",
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


def loss_wasserstein(y_human, y_llm):
    return scipy.stats.wasserstein_distance(y_human, y_llm)


def transform_identity(y_human, y_llm, y_llm_test):
    return y_llm_test


def transform_constant(y_human, y_llm, y_llm_test):
    b = statistics.mean(h - m for h, m in zip(y_human, y_llm))
    return [y + b for y in y_llm_test]


def transform_linear(y_human, y_llm, y_llm_test):
    a = sum(h * m for h, m in zip(y_human, y_llm)) / sum(m * m for m in y_llm)
    return [a * y for y in y_llm_test]


def _ols(y_human, y_llm):
    mean_h, mean_m = statistics.mean(y_human), statistics.mean(y_llm)
    var_m = sum((m - mean_m) ** 2 for m in y_llm)
    if var_m == 0:
        return 0.0, mean_h
    a = sum((m - mean_m) * (h - mean_h) for m, h in zip(y_llm, y_human)) / var_m
    return a, mean_h - a * mean_m


def transform_affine(y_human, y_llm, y_llm_test):
    a, b = _ols(y_human, y_llm)
    return [a * y + b for y in y_llm_test]


def transform_minmax(y_human, y_llm, y_llm_test):
    min_h, max_h = min(y_human), max(y_human)
    min_m, max_m = min(y_llm), max(y_llm)
    if max_m - min_m == 0:
        return y_llm_test
    return [(y - min_m) / (max_m - min_m) * (max_h - min_h) + min_h for y in y_llm_test]


def transform_musigma(y_human, y_llm, y_llm_test):
    mean_h, std_h = statistics.mean(y_human), statistics.stdev(y_human)
    mean_m, std_m = statistics.mean(y_llm), statistics.stdev(y_llm)
    if std_m == 0:
        return y_llm_test
    return [(y - mean_m) / std_m * std_h + mean_h for y in y_llm_test]


def transform_wasserstein_affine(y_human, y_llm, y_llm_test):
    def loss(params):
        a, b = params
        return loss_wasserstein(y_human, [a * y + b for y in y_llm])

    b_init = statistics.mean(h - m for h, m in zip(y_human, y_llm))
    result = scipy.optimize.minimize(loss, [1.0, b_init], method="Nelder-Mead")
    a, b = result.x
    return [a * y + b for y in y_llm_test]


METHODS = [
    ("Identity", transform_identity),
    ("MSE Constant", transform_constant),
    ("MSE Linear", transform_linear),
    ("MSE Affine", transform_affine),
    ("Wass. Affine", transform_wasserstein_affine),
    ("min/max-matching", transform_minmax),
    ("mu/sigma-matching", transform_musigma),
]


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


def score_anchor(human, scores, indices, transform, held_out):
    anchor_human = [human[i] for i in indices]
    anchor_scores = [scores[i] for i in indices]
    eval_idx = [i for i in range(len(scores)) if i not in set(indices)] if held_out else range(len(scores))
    eval_human = [human[i] for i in eval_idx]
    eval_scores = [scores[i] for i in eval_idx]
    transformed = np.clip(transform(anchor_human, anchor_scores, eval_scores), 0, 10).tolist()
    return loss_wasserstein(eval_human, transformed)


def main(args):
    rng = np.random.default_rng(args.seed)
    common, human, agg = load_data(args.judges, args.judge_scores_file)
    print(f"n_conversations={len(common)} judges={args.judges} held_out={args.held_out}\n")

    header = f"{'Method':<20} {'rand-8':>7} {'rand-32':>7} {'rand-128':>8} {'opt-8':>7} {'opt-32':>8} {'opt-128':>8}"
    print(header)
    for name, transform in METHODS:
        row = [name]
        for size in (8, 32, 128):
            random_losses = []
            for _ in range(10):
                indices = random_anchor(len(human), size, rng)
                for scores in agg.values():
                    random_losses.append(score_anchor(human, scores, indices, transform, args.held_out))
            row.append(statistics.mean(random_losses))
        for size in (8, 32, 128):
            optimized_losses = []
            for judge, scores in agg.items():
                others = [s for j, s in agg.items() if j != judge]
                indices = optimized_anchor(human, others, size, transform, rng)
                optimized_losses.append(score_anchor(human, scores, indices, transform, args.held_out))
            row.append(statistics.mean(optimized_losses))
        print(f"{row[0]:<20} {row[1]:>7.3f} {row[2]:>7.3f} {row[3]:>8.3f} {row[4]:>7.3f} {row[5]:>8.3f} {row[6]:>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=ORIGINAL_JUDGES)
    parser.add_argument("--judge_scores_file", default=str(CALIBRATION_DIR / "judge_scores_dataset.jsonl"))
    parser.add_argument("--held_out", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
