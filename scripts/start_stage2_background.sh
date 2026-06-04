#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p logs checkpoints/stage2

export PYTHONUNBUFFERED=1

nohup python train.py \
  --stage 2 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage2/ \
  --checkpoint checkpoints/stage1/best.pt \
  --reset_optimizer_state \
  --batch_size 16 \
  --grad_accum_steps 1 \
  --precision "${TRAIN_PRECISION:-bf16}" \
  --require_cuda \
  --no_compile \
  "$@" \
  > logs/stage2.out 2>&1 &

echo "$!" > logs/stage2.pid
echo "started stage2 pid=$(cat logs/stage2.pid)"
echo "log: logs/stage2.out"
echo "csv: checkpoints/stage2/training_log.csv"
