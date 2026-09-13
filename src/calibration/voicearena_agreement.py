import argparse
import json
from itertools import combinations
from pathlib import Path

from scipy.stats import wasserstein_distance

from multi_judge_fit import apply_calibration

CALIBRATION_DIR = Path(__file__).resolve().parent


def load_voicearena_scores(path):
    scores = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            scores.setdefault(row["judge"], []).append(row["score"])
    return scores


def main(args):
    calibration = json.loads(Path(args.calibration_file).read_text())["calibrations"]
    scores_by_judge = load_voicearena_scores(args.voicearena_scores_file)

    judges = [j for j in calibration if j in scores_by_judge]
    raw = {j: scores_by_judge[j] for j in judges}
    calib = {j: [apply_calibration(calibration[j], s) for s in scores_by_judge[j]] for j in judges}

    print(f"judges: {judges}\n")
    for j in judges:
        r, c = raw[j], calib[j]
        print(f"  {j:36s} raw mean={sum(r)/len(r):6.3f}  calib mean={sum(c)/len(c):6.3f}")

    raw_dists = [wasserstein_distance(raw[a], raw[b]) for a, b in combinations(judges, 2)]
    calib_dists = [wasserstein_distance(calib[a], calib[b]) for a, b in combinations(judges, 2)]

    print(f"\npairwise Wasserstein distance across {len(judges)} judges ({len(raw_dists)} pairs):")
    print(f"  raw:        mean={sum(raw_dists)/len(raw_dists):.4f}  max={max(raw_dists):.4f}")
    print(f"  calibrated: mean={sum(calib_dists)/len(calib_dists):.4f}  max={max(calib_dists):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_file", default=str(CALIBRATION_DIR / "multi_judge_candor_calibration.json"))
    parser.add_argument("--voicearena_scores_file", default=str(CALIBRATION_DIR / "voicearena_success_scores.jsonl"))
    main(parser.parse_args())
