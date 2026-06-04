#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p logs checkpoints/stage3

export PYTHONUNBUFFERED=1

nohup python train.py \
  --stage 3 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage3/ \
  --checkpoint checkpoints/stage2/best.pt \
  --reset_optimizer_state \
  --tokenizer_path tokenizer/tokenizer.model \
  --precision "${TRAIN_PRECISION:-bf16}" \
  --require_cuda \
  --no_compile \
  "$@" \
  > logs/stage3.out 2>&1 &

echo "$!" > logs/stage3.pid
echo "started stage3 pid=$(cat logs/stage3.pid)"
echo "log: logs/stage3.out"
echo "csv: checkpoints/stage3/training_log.csv"
