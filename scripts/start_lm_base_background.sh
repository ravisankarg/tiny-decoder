#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec scripts/start_lm_experiment_background.sh model_config_40m.json lm_40m "$@"
