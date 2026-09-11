"""Fit the mu/sigma-affine CANDOR calibration (fit_candor_calibration.py) for several judges
at once, on the same anchor set, so their calibrated scores share one scale and can be pooled.
"""

import argparse
import json
from pathlib import Path

from fit_candor_calibration import fit, load_ground_truth, load_judge_scores

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration_data"

JUDGES = ["phi4multimodal_CoT_summary_speech", "qwen25omni_success_text"]


def main(args):
    ground_truth = load_ground_truth(CALIBRATION_DIR / "pointwise_dataset.jsonl")
    anchor_ids = Path(args.anchor_ids_file).read_text().split()

    calibrations = {}
    for judge in args.judges:
        judge_scores = load_judge_scores(CALIBRATION_DIR / "judge_scores_dataset.jsonl", judge)
        missing = [c for c in anchor_ids if c not in judge_scores or c not in ground_truth]
        if missing:
            raise ValueError(f"{judge}: {len(missing)} anchor ids missing score or ground truth")
        params = fit(anchor_ids, judge_scores, ground_truth)
        calibrations[judge] = params
        print(f"{judge}: judge mean={params['mean_judge']:.3f} std={params['std_judge']:.3f}"
              f"  ->  human mean={params['mean_human']:.3f} std={params['std_human']:.3f}"
              f"  (slope c={params['std_human']/params['std_judge']:.4f})")

    out_path = Path(args.output_path)
    out_path.write_text(json.dumps({"anchor_ids_file": str(args.anchor_ids_file), "calibrations": calibrations}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=JUDGES)
    parser.add_argument("--anchor_ids_file", default=str(CALIBRATION_DIR / "candor_anchor_ids_32.txt"))
    parser.add_argument("--output_path", default=str(CALIBRATION_DIR / "multi_judge_candor_calibration.json"))
    main(parser.parse_args())
