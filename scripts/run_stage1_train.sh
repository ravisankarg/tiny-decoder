#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p logs checkpoints/stage1
export PYTHONUNBUFFERED=1
echo "$$" > logs/stage1.pid

checkpoint_args=()
if [[ -f checkpoints/stage1/latest.pt ]]; then
  checkpoint_args=(--checkpoint checkpoints/stage1/latest.pt)
elif [[ -f checkpoints/stage1/best.pt ]]; then
  checkpoint_args=(--checkpoint checkpoints/stage1/best.pt)
fi

exec python train.py \
  --stage 1 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage1/ \
  "${checkpoint_args[@]}" \
  --batch_size 32 \
  --grad_accum_steps 1 \
  --fp16 \
  --require_cuda \
  --no_compile \
  "$@"
