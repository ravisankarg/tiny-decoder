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

checkpoint_args=()
if [[ -f "checkpoints/${experiment_name}/latest.pt" ]]; then
  checkpoint_args=(--checkpoint "checkpoints/${experiment_name}/latest.pt")
elif [[ -f "checkpoints/${experiment_name}/best.pt" ]]; then
  checkpoint_args=(--checkpoint "checkpoints/${experiment_name}/best.pt")
fi

nohup python train.py \
  --stage 0 \
  --data_dir data/ \
  --config_path "$config_path" \
  --output_dir "checkpoints/${experiment_name}/" \
  --experiment_name "$experiment_name" \
  "${checkpoint_args[@]}" \
  --epochs "${LM_EPOCHS:-3}" \
  --batch_size "${LM_BATCH_SIZE:-8}" \
  --grad_accum_steps "${LM_GRAD_ACCUM_STEPS:-2}" \
  --lr "${LM_LR:-2e-4}" \
  --warmup_steps "${LM_WARMUP_STEPS:-1000}" \
  --fp16 \
  --require_cuda \
  --no_compile \
  "$@" \
  > "logs/${experiment_name}.out" 2>&1 &

echo "$!" > "logs/${experiment_name}.pid"
echo "started ${experiment_name} pid=$(cat "logs/${experiment_name}.pid")"
echo "log: logs/${experiment_name}.out"
echo "csv: checkpoints/${experiment_name}/training_log.csv"
echo "metadata: checkpoints/${experiment_name}/metadata.json"
