"""Fit the mu/sigma-affine calibration function (paper section "Calibrating Future Judges")
on CANDOR: maps a judge's raw pointwise scores onto the human PCS ground-truth scale,
using a small anchor set of conversations with both a judge score and a true label.

    t(s) = (s - mean(anchor judge scores)) / std(anchor judge scores)
               * std(anchor human scores) + mean(anchor human scores)

This is monotonic (assuming the anchor judge scores aren't constant), so it preserves
rank correlations -- only the scale changes. Inputs are calibration_data/pointwise_dataset.jsonl
(ground truth) and calibration_data/judge_scores_dataset.jsonl (raw judge scores); both were
exported from the CANDOR pointwise pipeline (see build_pointwise_dataset.py / build_judge_scores_dataset.py).
"""

import argparse
import json
import math
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration_data"


def _mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, std


def load_ground_truth(path):
    """CANDOR rows only, excluding the mock VoiceArena placeholders."""
    gt = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["conversation"].startswith("CANDOR") and not row["mock"]:
                gt[row["conversation"]] = row["score"]
    return gt


def load_judge_scores(path, judge):
    scores = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["judge"] == judge:
                scores[row["conversation"]] = row["score"]
    return scores


def spearman(x, y):
    def rankdata(vals):
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy)


def fit(anchor_ids, judge_scores, ground_truth):
    anchor_judge = [judge_scores[c] for c in anchor_ids]
    anchor_gt = [ground_truth[c] for c in anchor_ids]
    mean_j, std_j = _mean_std(anchor_judge)
    mean_h, std_h = _mean_std(anchor_gt)
    if std_j == 0:
        raise ValueError("anchor judge scores have zero variance -- cannot fit mu/sigma affine")
    return {"mean_judge": mean_j, "std_judge": std_j, "mean_human": mean_h, "std_human": std_h}


def apply_calibration(params, score):
    return (score - params["mean_judge"]) / params["std_judge"] * params["std_human"] + params["mean_human"]


def main(args):
    ground_truth = load_ground_truth(CALIBRATION_DIR / "pointwise_dataset.jsonl")
    judge_scores = load_judge_scores(CALIBRATION_DIR / "judge_scores_dataset.jsonl", args.judge)

    anchor_ids = Path(args.anchor_ids_file).read_text().split()
    missing = [c for c in anchor_ids if c not in judge_scores or c not in ground_truth]
    if missing:
        raise ValueError(f"{len(missing)} anchor ids missing judge score or ground truth: {missing[:5]}")

    params = fit(anchor_ids, judge_scores, ground_truth)
    print(f"[fit] anchor n={len(anchor_ids)} judge={args.judge}")
    print(f"  judge:  mean={params['mean_judge']:.3f} std={params['std_judge']:.3f}")
    print(f"  human:  mean={params['mean_human']:.3f} std={params['std_human']:.3f}")

    common = sorted(c for c in judge_scores if c in ground_truth)
    raw = [judge_scores[c] for c in common]
    calibrated = [apply_calibration(params, s) for s in raw]
    true = [ground_truth[c] for c in common]

    rho_raw = spearman(raw, true)
    rho_calibrated = spearman(calibrated, true)
    raw_mean, raw_std = _mean_std(raw)
    calib_mean, calib_std = _mean_std(calibrated)
    true_mean, true_std = _mean_std(true)

    print(f"\n[full set] n={len(common)}")
    print(f"  spearman raw={rho_raw:.3f} calibrated={rho_calibrated:.3f} (identical up to fp error -- monotonic transform)")
    print(f"  raw score:        mean={raw_mean:.3f} std={raw_std:.3f}")
    print(f"  calibrated score: mean={calib_mean:.3f} std={calib_std:.3f}")
    print(f"  human score:      mean={true_mean:.3f} std={true_std:.3f}")

    output = {
        "judge": args.judge,
        "method": "mu_sigma_affine",
        "anchor_ids_file": str(args.anchor_ids_file),
        "anchor_size": len(anchor_ids),
        "anchor_ids": anchor_ids,
        "params": params,
        "diagnostics": {
            "n_full": len(common),
            "spearman_raw": rho_raw,
            "spearman_calibrated": rho_calibrated,
            "raw_mean": raw_mean, "raw_std": raw_std,
            "calibrated_mean": calib_mean, "calibrated_std": calib_std,
            "human_mean": true_mean, "human_std": true_std,
        },
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote calibration params to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="phi4multimodal_CoT_questions_speech")
    parser.add_argument("--anchor_ids_file", default=str(CALIBRATION_DIR / "candor_anchor_ids_32.txt"))
    parser.add_argument("--output_path", default=str(CALIBRATION_DIR / "phi_candor_calibration.json"))
    main(parser.parse_args())
