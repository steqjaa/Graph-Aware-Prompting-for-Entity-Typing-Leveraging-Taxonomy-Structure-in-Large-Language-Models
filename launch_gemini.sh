#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 validate_external_inputs.py
mkdir -p logs
python3 -u run_gemini_external.py --workers 20 --rpm 120 2>&1 | tee -a logs/gemini.log

python3 save_machine_results.py --model gemini
