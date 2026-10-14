# Final-study workflow

Use the root README and `docs/REPRODUCIBILITY.md` for current commands.

The original rerun plan is preserved at `archive/project_documentation_20261012/README_RERUNS.md`. It planned more child runs than the final observed study. Current scope is in `configs/final_study.json`.

Offline replay: `python scripts/reproduce_final.py`.
Packaging observed outcomes: `python scripts/package_results.py --run all --output final_results.zip`.
