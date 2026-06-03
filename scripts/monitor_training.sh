#!/usr/bin/env bash
set -euo pipefail

stage="${1:-1}"
cd "$(dirname "$0")/.."

pid_file="logs/stage${stage}.pid"
log_file="logs/stage${stage}.out"
csv_file="checkpoints/stage${stage}/training_log.csv"

if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "stage${stage}: running pid=$pid"
  else
    echo "stage${stage}: pid file exists but process is not running pid=$pid"
  fi
else
  echo "stage${stage}: no pid file"
fi

if [[ -f "$csv_file" ]]; then
  echo
  echo "latest metrics:"
  tail -n 5 "$csv_file"
fi

if [[ -f "$log_file" ]]; then
  echo
  echo "latest log:"
  tail -n 40 "$log_file"
fi
