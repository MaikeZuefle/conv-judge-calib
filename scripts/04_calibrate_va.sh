#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python src/calibration/voicearena_agreement.py \
  --calibration_file src/calibration/multi_judge_candor_calibration.json \
  --voicearena_scores_file src/calibration/voicearena_success_scores.jsonl
