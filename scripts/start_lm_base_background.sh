#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p logs checkpoints/lm_base

export PYTHONUNBUFFERED=1

checkpoint_args=()
if [[ -f checkpoints/lm_base/latest.pt ]]; then
  checkpoint_args=(--checkpoint checkpoints/lm_base/latest.pt)
elif [[ -f checkpoints/lm_base/best.pt ]]; then
  checkpoint_args=(--checkpoint checkpoints/lm_base/best.pt)
fi

nohup python train.py \
  --stage 0 \
  --data_dir data/ \
  --config_path model_config_40m.json \
  --output_dir checkpoints/lm_base/ \
  "${checkpoint_args[@]}" \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --fp16 \
  --require_cuda \
  --no_compile \
  "$@" \
  > logs/lm_base.out 2>&1 &

echo "$!" > logs/lm_base.pid
echo "started lm_base pid=$(cat logs/lm_base.pid)"
echo "log: logs/lm_base.out"
echo "csv: checkpoints/lm_base/training_log.csv"
