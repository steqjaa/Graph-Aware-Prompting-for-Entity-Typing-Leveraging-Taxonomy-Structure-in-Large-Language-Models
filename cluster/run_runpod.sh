#!/usr/bin/env bash
# Run within the pinned final vLLM GPU environment. Fixed seed on every retry.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 scripts/restore_resources.py
python3 validate_external_inputs.py
RUN_OUTPUT_ROOT=${RUN_OUTPUT_ROOT:-results/generated/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_OUTPUT_ROOT/logs"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then seeds=(0 42 123); fi
for seed in "${seeds[@]}"; do
  case "$seed" in 0) experiment=all ;; 42|123) experiment=main ;; *) echo 'Allowed seeds: 0 42 123' >&2; exit 1 ;; esac
  set +e
  python3 -u scripts/run_llama_external.py \
    --experiment "$experiment" --seed "$seed" --max-new-tokens 0 \
    --output-root "$RUN_OUTPUT_ROOT/llama_seed_${seed}" \
    2>&1 | tee -a "$RUN_OUTPUT_ROOT/logs/llama_seed_${seed}.log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  if [ "${statuses[1]}" -ne 0 ]; then exit "${statuses[1]}"; fi
  case "${statuses[0]}" in
    0) echo "Seed $seed completed successfully." ;;
    2) echo "Seed $seed retained unsuccessful responses; continuing without changing its seed." ;;
    *) exit "${statuses[0]}" ;;
  esac
done
