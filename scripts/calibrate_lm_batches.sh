#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

export PYTHONUNBUFFERED=1
export LM_DATASET_NAME="${LM_DATASET_NAME:-lm_prose}"

configs=(
  "model_config_10m.json:lm_10m:40 44 48 52 56"
  "model_config.json:lm_20m:24 28 32 36 40"
  "model_config_30m.json:lm_30m:10 12 13 14 16"
  "model_config_40m.json:lm_40m:4 6 7 8 10"
)

mkdir -p logs/batch_calibration /tmp/tiny_decoder_batch_probe

for item in "${configs[@]}"; do
  IFS=: read -r config experiment candidates <<< "$item"
  best=""
  echo "== ${experiment} (${config}) =="
  for batch in $candidates; do
    out_dir="/tmp/tiny_decoder_batch_probe/${experiment}_bs${batch}"
    log_path="logs/batch_calibration/${experiment}_bs${batch}.log"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    echo "testing batch_size=${batch}"
    if python train.py \
      --stage 0 \
      --data_dir data/ \
      --config_path "$config" \
      --output_dir "$out_dir" \
      --experiment_name "${experiment}_probe_bs${batch}" \
      --epochs 1 \
      --batch_size "$batch" \
      --grad_accum_steps 1 \
      --lr 2e-4 \
      --warmup_steps 10 \
      --fp16 \
      --require_cuda \
      --no_compile \
      --overwrite \
      --max_steps "${LM_CALIBRATE_STEPS:-2}" \
      > "$log_path" 2>&1; then
      best="$batch"
      echo "ok batch_size=${batch}"
    else
      echo "failed batch_size=${batch}; see ${log_path}"
      break
    fi
  done
  if [[ -z "$best" ]]; then
    echo "${experiment}: no candidate passed"
  else
    echo "${experiment}: best_batch_size=${best}"
  fi
done
