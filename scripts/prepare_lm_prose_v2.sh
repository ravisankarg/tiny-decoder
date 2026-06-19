#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p data/.hf_home data/.hf_datasets_cache data/.hf_hub_cache
export HF_HOME="${HF_HOME:-$PWD/data/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/data/.hf_datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/data/.hf_hub_cache}"

# Large streaming parquet shards can exceed Hugging Face's short default HTTP
# timeout on normal home connections. Keep these overrideable from the shell.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

# Full target for the next 40M-only run. Override this for smoke tests, e.g.
# LM_PROSE_V2_TARGET_TOKENS=2000000 scripts/prepare_lm_prose_v2.sh --overwrite
export LM_PROSE_V2_TARGET_TOKENS="${LM_PROSE_V2_TARGET_TOKENS:-1200000000}"
export LM_PROSE_V2_MIN_WORDS="${LM_PROSE_V2_MIN_WORDS:-40}"

# Recommended v2 mix:
# 45% FineWeb-Edu, 25% Wikipedia, 10% MS MARCO passages, 10% app descriptions,
# 5% OpenWebMath-style prose, 5% clean QA prose.
export LM_PROSE_V2_FINEWEB_EDU_RATIO="${LM_PROSE_V2_FINEWEB_EDU_RATIO:-0.45}"
export LM_PROSE_V2_WIKIPEDIA_RATIO="${LM_PROSE_V2_WIKIPEDIA_RATIO:-0.25}"
export LM_PROSE_V2_MSMARCO_RATIO="${LM_PROSE_V2_MSMARCO_RATIO:-0.10}"
export LM_PROSE_V2_APP_DESC_RATIO="${LM_PROSE_V2_APP_DESC_RATIO:-0.10}"
export LM_PROSE_V2_OPENWEBMATH_RATIO="${LM_PROSE_V2_OPENWEBMATH_RATIO:-0.05}"
export LM_PROSE_V2_QA_RATIO="${LM_PROSE_V2_QA_RATIO:-0.05}"

# Optional source-specific document caps. 0 means no explicit cap; the token
# quota stops streaming when enough accepted text has been collected.
export FINEWEB_EDU_MAX_DOCS="${FINEWEB_EDU_MAX_DOCS:-0}"
export WIKIPEDIA_MAX_DOCS="${WIKIPEDIA_MAX_DOCS:-0}"
export MSMARCO_PASSAGE_MAX_DOCS="${MSMARCO_PASSAGE_MAX_DOCS:-300000}"
export APP_DESC_MAX_DOCS="${APP_DESC_MAX_DOCS:-100000}"
export OPENWEBMATH_MAX_DOCS="${OPENWEBMATH_MAX_DOCS:-0}"
export QA_PROSE_MAX_DOCS="${QA_PROSE_MAX_DOCS:-0}"

# Dataset IDs/configs can be overridden if a mirror or newer snapshot is needed.
export FINEWEB_EDU_CONFIG="${FINEWEB_EDU_CONFIG:-sample-10BT}"
export WIKIPEDIA_CONFIG="${WIKIPEDIA_CONFIG:-20231101.en}"
export OPENWEBMATH_DATASET="${OPENWEBMATH_DATASET:-open-web-math/open-web-math}"
export OPENWEBMATH_SPLIT="${OPENWEBMATH_SPLIT:-train}"

python prepare_datasets.py \
  --stage lm_prose_v2 \
  --output_dir data \
  --tokenizer_path tokenizer/tokenizer.model \
  --max_seq_len 256 \
  "$@"
