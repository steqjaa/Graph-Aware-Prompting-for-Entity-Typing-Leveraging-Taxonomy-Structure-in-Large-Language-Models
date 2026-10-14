# Historical final experiment — 9 October 2026

Start with docs/FINAL_RUN_AND_EVALUATION.md for the exact generation settings and evaluation command. Read reports/final_comparison/GRAPH_CONTEXT_FOCUS.md for the primary graph-aware versus no-graph results, reports/final_comparison/entity_distribution/ENTITY_TABLES.md for the entity composition, and reports/final_comparison/REPORT.md for the supporting comparisons and reports/final_comparison/FINDINGS.md for the interpretation.

The final Llama inference code is scripts/run_llama_json_mode.py plus scripts/retry_llama_fresh_seed.py. scripts/run_llama_fast.py is the earlier runner retained for its shared audit functions and tests; its bounded-schema mode was not used for the final experiment. Original submitted COMA code and logs are archived in archive/final_coma_run.

Final result locations:
- results/gemini_1k_fresh: 9,000 Gemini responses.
- results/json_mode_final/llama_1k_fast: 9,000 Llama responses.
- results/json_mode_final/llama_children_removed_fast: 1,353 Llama child-removal responses.

The original README and older guides describe prior stages. Use README_FINAL.md and FINAL_RUN_AND_EVALUATION.md for the final experiment. No inference rerun is required to reproduce the scores and figures.
