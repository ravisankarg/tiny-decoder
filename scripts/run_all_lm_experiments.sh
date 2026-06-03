#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_one() {
  local config="$1"
  local experiment="$2"
  echo "== starting ${experiment} =="
  scripts/start_lm_experiment_background.sh "$config" "$experiment" --overwrite
  local pid
  pid="$(cat "logs/${experiment}.pid")"
  while ps -p "$pid" > /dev/null 2>&1; do
    sleep 60
  done
  if ! tail -n 20 "logs/${experiment}.out" | grep -q "=== DONE: train.py ==="; then
    echo "ERROR: ${experiment} did not finish cleanly. See logs/${experiment}.out" >&2
    exit 1
  fi
  echo "== finished ${experiment} =="
}

run_one model_config_10m.json lm_10m
run_one model_config.json lm_20m
run_one model_config_30m.json lm_30m
run_one model_config_40m.json lm_40m

scripts/compare_lm_experiments.py
