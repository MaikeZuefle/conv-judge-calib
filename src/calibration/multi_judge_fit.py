import argparse
import json
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parent

JUDGES = [
    "phi4mm_speech",
    "phi4mm_text",
    "qwen3omni_speech",
    "qwen3omni_text",
    "qwen35_text",
    "qwen25omni_success_speech",
    "qwen25omni_success_text",
]


def load_ground_truth(path):
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


def _mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, std


def fit(anchor_ids, judge_scores, ground_truth):
    anchor_judge = [judge_scores[c] for c in anchor_ids]
    anchor_gt = [ground_truth[c] for c in anchor_ids]
    mean_j, std_j = _mean_std(anchor_judge)
    mean_h, std_h = _mean_std(anchor_gt)
    if std_j == 0:
        raise ValueError("anchor judge scores have zero variance -- cannot fit mu/sigma affine")
    slope = std_h / std_j
    return {"slope": slope, "intercept": mean_h - slope * mean_j}


def fit_wasserstein_affine(anchor_ids, judge_scores, ground_truth):
    from scipy.optimize import minimize
    from scipy.stats import wasserstein_distance

    xs = [judge_scores[c] for c in anchor_ids]
    ys = [ground_truth[c] for c in anchor_ids]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs) / len(xs)
    if var_x == 0:
        raise ValueError("anchor judge scores have zero variance -- cannot fit Wasserstein affine")

    b_init = mean_y - mean_x
    result = minimize(
        lambda p: wasserstein_distance(ys, [p[0] * x + p[1] for x in xs]),
        x0=[1.0, b_init],
        method="Nelder-Mead",
    )
    slope, intercept = result.x
    return {"slope": slope, "intercept": intercept}


def apply_calibration(params, score):
    return params["slope"] * score + params["intercept"]


FIT_METHODS = {"mu_sigma_affine": fit, "wasserstein_affine": fit_wasserstein_affine}


def main(args):
    ground_truth = load_ground_truth(CALIBRATION_DIR / "pointwise_dataset.jsonl")
    anchor_ids = Path(args.anchor_ids_file).read_text().split()
    fit_method = FIT_METHODS[args.method]

    calibrations = {}
    for judge in args.judges:
        judge_scores = load_judge_scores(args.judge_scores_file, judge)
        usable = [c for c in anchor_ids if c in judge_scores and c in ground_truth]
        if len(usable) < len(anchor_ids):
            print(f"[PARTIAL] {judge}: using {len(usable)}/{len(anchor_ids)} anchor ids "
                  f"({len(anchor_ids) - len(usable)} missing score or ground truth)")
        if len(usable) < 2:
            print(f"[SKIP] {judge}: only {len(usable)} usable anchor ids, cannot fit")
            continue
        params = fit_method(usable, judge_scores, ground_truth)
        calibrations[judge] = params
        print(f"{judge}: slope={params['slope']:.4f} intercept={params['intercept']:.4f}")

    out_path = Path(args.output_path)
    out_path.write_text(json.dumps({"method": args.method, "anchor_ids_file": str(args.anchor_ids_file),
                                     "calibrations": calibrations}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=JUDGES)
    parser.add_argument("--judge_scores_file", default=str(CALIBRATION_DIR / "judge_scores_dataset.jsonl"))
    parser.add_argument("--method", choices=list(FIT_METHODS), default="mu_sigma_affine")
    parser.add_argument("--anchor_ids_file", default=str(CALIBRATION_DIR / "candor_anchor_ids_32.txt"))
    parser.add_argument("--output_path", default=str(CALIBRATION_DIR / "multi_judge_candor_calibration.json"))
    main(parser.parse_args())
