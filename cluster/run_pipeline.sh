#!/usr/bin/env bash
# For a node/environment with BOTH GPU and analysis dependencies installed.
set -euo pipefail
cd "$(dirname "$0")/.."
bash cluster/run_fast.sh "$@"
python scripts/evaluate_all.py
