#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 validate_external_inputs.py
mkdir -p logs
for rerun_seed in 0 42 123; do
  apptainer exec --cleanenv --nv \
    --env HF_HOME="$HOME/hf_cache" \
    --env CC="$HOME/thesis-compiler/bin/x86_64-conda-linux-gnu-gcc" \
    --env CXX="$HOME/thesis-compiler/bin/x86_64-conda-linux-gnu-g++" \
    --env VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$HOME/containers/pytorch.sif" \
    "$HOME/thesis-vllm-env/bin/python" -u \
    scripts/run_llama_external.py \
    --experiment all --seed "$rerun_seed" --max-new-tokens 0 \
    --output-root "results/external_demo/llama_seed_${rerun_seed}" \
    2>&1 | tee -a "logs/llama_seed_${rerun_seed}.log"
done

python3 save_machine_results.py --model llama
