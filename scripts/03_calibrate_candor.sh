#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python src/calibration/multi_judge_fit.py \
  --judges qwen25omni_success_speech qwen25omni_success_text \
  --method mu_sigma_affine \
  --output_path src/calibration/multi_judge_candor_calibration.json