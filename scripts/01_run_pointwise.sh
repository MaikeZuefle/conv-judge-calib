#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python src/run_judge.py \
  --dataset candor \
  --model qwen25omni \
  --prompt success \
  --input_modality speech \
  --output_folder outputs

python src/eval_judge.py \
  --dataset candor \
  --model qwen25omni \
  --prompt success \
  --input_modality speech \
  --output_folder outputs
