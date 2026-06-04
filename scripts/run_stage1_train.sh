#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

export PYTHONUNBUFFERED=1

overwrite=0
for arg in "$@"; do
  if [[ "$arg" == "--overwrite" ]]; then
    overwrite=1
  fi
done

if (( overwrite == 1 )); then
  rm -rf checkpoints/stage1
  rm -f logs/stage1.out logs/stage1.pid
fi
mkdir -p logs checkpoints/stage1
echo "$$" > logs/stage1.pid

checkpoint_args=()
if (( overwrite == 0 )); then
  if [[ -f checkpoints/stage1/latest.pt ]]; then
    checkpoint_args=(--checkpoint checkpoints/stage1/latest.pt)
  elif [[ -f checkpoints/stage1/best.pt ]]; then
    checkpoint_args=(--checkpoint checkpoints/stage1/best.pt)
  fi
fi

exec python train.py \
  --stage 1 \
  --data_dir data/ \
  --config_path model_config.json \
  --output_dir checkpoints/stage1/ \
  "${checkpoint_args[@]}" \
  --batch_size 32 \
  --grad_accum_steps 1 \
  --precision "${TRAIN_PRECISION:-bf16}" \
  --require_cuda \
  --no_compile \
  "$@"
