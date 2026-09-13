#!/bin/bash
set -e
cd "$(dirname "$0")/.."

BATCHES_FILE=path/to/your_batches.jsonl  # list of pairwise comparisons to make

python src/run_bws_judge.py \
  --batches_file "$BATCHES_FILE" \
  --output_file outputs/pairwise_qwen25omni.jsonl \
  --model qwen25omni \
  --layout closing_rank \
  --transcript_source candor \
  --parse_as rank
