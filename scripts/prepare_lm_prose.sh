#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

python prepare_datasets.py \
  --stage lm_prose \
  --output_dir data \
  --tokenizer_path tokenizer/tokenizer.model \
  --max_seq_len 256 \
  "$@"
