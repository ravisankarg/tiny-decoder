#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set. Private or gated datasets may fail."
fi

python prepare_datasets.py --stage 1_text --output_dir data/

python build_tokenizer.py \
  --corpus_txt data/corpus.txt \
  --vocab_size 16000 \
  --output_dir tokenizer/

python prepare_datasets.py \
  --stage 1 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model

python prepare_datasets.py \
  --stage 2 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model

python prepare_datasets.py \
  --stage 3 \
  --output_dir data/ \
  --tokenizer_path tokenizer/tokenizer.model

echo "=== DONE: prepare_all_datasets.sh ==="
