#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

config_path="${1:-}"
experiment_name="${2:-}"
shift 2 || true

if [[ -z "$config_path" || -z "$experiment_name" ]]; then
  echo "usage: scripts/start_lm_experiment_background.sh <config_path> <experiment_name> [train.py args...]"
  echo "example: scripts/start_lm_experiment_background.sh model_config_40m.json lm_40m --overwrite"
  exit 2
fi

mkdir -p logs "checkpoints/${experiment_name}"

export PYTHONUNBUFFERED=1
export LM_DATASET_NAME="${LM_DATASET_NAME:-lm_prose}"

param_count="$(python - "$config_path" <<'PY'
import json
import sys
from model_config import TinyDecoderConfig, estimate_param_count
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = TinyDecoderConfig(**json.load(handle))
print(estimate_param_count(cfg))
PY
)"

if [[ -z "${LM_BATCH_SIZE:-}" || -z "${LM_GRAD_ACCUM_STEPS:-}" ]]; then
  if (( param_count <= 9000000 )); then
    auto_batch_size=32
    auto_grad_accum_steps=2
  elif (( param_count <= 21000000 )); then
    auto_batch_size=16
    auto_grad_accum_steps=4
  elif (( param_count <= 32000000 )); then
    auto_batch_size=8
    auto_grad_accum_steps=8
  else
    auto_batch_size=4
    auto_grad_accum_steps=16
  fi
fi

batch_size="${LM_BATCH_SIZE:-$auto_batch_size}"
grad_accum_steps="${LM_GRAD_ACCUM_STEPS:-$auto_grad_accum_steps}"
precision="${LM_PRECISION:-${TRAIN_PRECISION:-bf16}}"
effective_tokens=$((batch_size * grad_accum_steps * 256))

checkpoint_args=()
if [[ -f "checkpoints/${experiment_name}/latest.pt" ]]; then
  checkpoint_args=(--checkpoint "checkpoints/${experiment_name}/latest.pt")
elif [[ -f "checkpoints/${experiment_name}/best.pt" ]]; then
  checkpoint_args=(--checkpoint "checkpoints/${experiment_name}/best.pt")
fi

if [[ "${LM_DRY_RUN:-0}" == "1" ]]; then
  echo "experiment: ${experiment_name}"
  echo "config: ${config_path}"
  echo "params: ${param_count}"
  echo "dataset: ${LM_DATASET_NAME}"
  echo "batch_size: ${batch_size}"
  echo "grad_accum_steps: ${grad_accum_steps}"
  echo "precision: ${precision}"
  echo "effective_tokens_per_update: ${effective_tokens}"
  exit 0
fi

nohup python train.py \
  --stage 0 \
  --data_dir data/ \
  --config_path "$config_path" \
  --output_dir "checkpoints/${experiment_name}/" \
  --experiment_name "$experiment_name" \
  "${checkpoint_args[@]}" \
  --epochs "${LM_EPOCHS:-3}" \
  --batch_size "$batch_size" \
  --grad_accum_steps "$grad_accum_steps" \
  --lr "${LM_LR:-2e-4}" \
  --warmup_steps "${LM_WARMUP_STEPS:-1000}" \
  --precision "$precision" \
  --require_cuda \
  --no_compile \
  "$@" \
  > "logs/${experiment_name}.out" 2>&1 &

echo "$!" > "logs/${experiment_name}.pid"
echo "started ${experiment_name} pid=$(cat "logs/${experiment_name}.pid")"
echo "params: ${param_count}"
echo "batch_size: ${batch_size}"
echo "grad_accum_steps: ${grad_accum_steps}"
echo "precision: ${precision}"
echo "effective_tokens_per_update: ${effective_tokens}"
echo "log: logs/${experiment_name}.out"
echo "csv: checkpoints/${experiment_name}/training_log.csv"
echo "metadata: checkpoints/${experiment_name}/metadata.json"
