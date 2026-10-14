#!/usr/bin/env bash
# Run INSIDE an allocated GPU node, with the vLLM environment activated.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
python -u scripts/run_llama_fast.py "$@" 2>&1 | tee -a logs/llama_fast.log
