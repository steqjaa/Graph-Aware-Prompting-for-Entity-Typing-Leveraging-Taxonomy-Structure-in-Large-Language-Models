# Final thesis experiments — 9 October 2026

The final comparison uses results/gemini_1k_fresh (9,000 responses), results/json_mode_final/llama_1k_fast (9,000), and results/json_mode_final/llama_children_removed_fast (1,353). Do not pool the older free-decoding, bounded-schema, pilot, or incomplete Gemini runs.

The 1,000-entity main experiment has nine conditions. Child removal covers 451 child-present entities under three taxonomy methods; the matching with-children responses come from the main run. Gemini child-removal inference was not supplied: the child-removal causal contrast is Llama only.

## Reproduce evaluation

From the project root, install requirements-analysis.txt and run:

```bash
python3 scripts/evaluate_all.py --llama results/json_mode_final/llama_1k_fast --children results/json_mode_final/llama_children_removed_fast
```

Scores and figures are written beneath each result directory's analysis folder. Combined tables and the report are in reports/final_comparison.

## Actual generation settings

Gemini: recorded provider model gemini-3.8-flash; temperature 1.0; application/json MIME mode; no response schema or explicit maxOutputTokens; provider-default thinking. Provider defaults are not unlimited generation.

Llama: pinned meta-llama/Llama-3.1-8B-Instruct revision 0e9e39f249a16976918f6564b8830bc894c89659; vLLM 0.31.0; H200 NVL; bfloat16; temperature 1.0; top_p 1; JSON-object decoding without a field schema; 32,768-token total context; output may use all remaining context; batches of 128, up to 32 simultaneous sequences. No explanation-character, label-character, or type-count caps. run_llama_json_mode.py is the actual runner. --structured-json must not be supplied for this final experiment.

```bash
apptainer exec --cleanenv --nv \
  --env HF_HOME="$HOME/hf_cache" \
  --env CC="$HOME/thesis-compiler/bin/x86_64-conda-linux-gnu-gcc" \
  --env CXX="$HOME/thesis-compiler/bin/x86_64-conda-linux-gnu-g++" \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  ~/containers/pytorch.sif \
  "$HOME/thesis-vllm-env/bin/python" -u \
  scripts/run_llama_json_mode.py --max-new-tokens 0 \
  --output-root results/json_mode_final
```

The base sampling seed is 0. Four persistent unsuccessful outputs (three main, one child-removal) were regenerated with seed 898122484 by retry_llama_fresh_seed.py. Their individual generation_config and retry_provenance record the override; the base run_config remains the original configuration. All archived attempts are retained. Retries select for format completion, so report the retry policy alongside the score comparison.

## Interpretation and validation

All final responses have STOP and pass the current nonempty-types/explanation parser. The scoring parser is applied uniformly to both models; no final empty type lists remain. The Gemini generation parser had allowed empty lists, while the evaluation parser requires nonempty lists. No predictions were manually repaired or invented.

The result audit verifies entity/condition identity, prompt/metadata hashes, model revision, matching configurations, and parsed predictions. Archived Llama source and parser hashes match the run configurations. Fifteen existing evaluation/runner tests pass.

Python JSON parsing uses the last value for duplicate object keys. Eighteen Llama responses (17 main, one child-removal) contain repeated keys; Gemini has zero. They remain included under the same parser policy and are listed in duplicate_json_keys.csv. Format completion does not imply good reasoning or correct classification. Manual-review worksheets are unannotated and require real human review.

Flat F1 uses normalized label sets. Hierarchical F1 uses ancestor closure, including self and roots, and is not LCA-F1. Taxonomy context is oracle context. Bootstrap confidence intervals use 2,000 paired entity resamples and are exploratory without multiplicity adjustment. This comparison concerns model systems: provider JSON mode, vLLM JSON decoding, seed, token budgets and thinking implementation cannot be made identical across providers.

## Figure values and dataset composition

Graph-aware versus no-graph prompting is the primary research contrast. GRAPH_CONTEXT_FOCUS.md contains all five required metrics and paired intervals. Model comparisons and child ablations are supporting analyses. Scores on plots use four decimal places; full-precision values remain in CSVs. Dense composition plots use numerically labelled heatmaps instead of unreadable narrow bars. Entity tables use recorded reference types and overlapping gold categories at depth 4; these are taxonomy groupings, not inferred domains.
