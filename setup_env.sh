#!/usr/bin/env bash
set -euo pipefail

python3.10 -m venv ./venv
source ./venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("WARNING: No GPU detected. Training will be extremely slow.")
    print("Use --device cpu for testing only. RTX 3050 required for full training.")
PY
echo "=== DONE: setup_env.sh ==="
