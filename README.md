# Graph-Aware Prompting for Entity Typing

This repository contains the frozen prompts, recorded model responses, and offline analysis for the final thesis study. The graph context is an oracle neighbourhood of an annotated class.

## Reproduce the final results without model calls

Use Python 3.12 and run from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/final-analysis.txt
python scripts/reproduce_final.py
```

This restores the exact full label inventory from the bundled checksum-pinned gzip, verifies frozen inputs and all recorded responses, replays the recovered analysis, checks five reference tables numerically and all 1,698 numeric appendix values, and generates the final diagnostics, qualitative review and figures under `results/derived/final/`. It makes **no model/API calls**. Allow several minutes and several GB of memory. Large raw result files are already included in this checkout.

Primary outputs:

| File | Meaning |
|---|---|
| `condition_summary.csv` | Completed-condition means and failure counts |
| `final_comparisons.csv` | Final matched graph/model/child effects, strata, intervals and paired denominators |
| `comparisons.csv` | Preserved earlier revision-core comparison replay for regression |
| `seed_summary.csv` | Common-cohort three-seed means and sample SD |
| `failure_sensitivity.csv` | Zero-utility and worst-case failure sensitivity |
| `extended_condition_summary.csv` | Micro/loose F1, strict accuracy, hierarchy consistency and hit rates |
| `model_gain_comparisons.csv` | Gemini versus each seed's graph gain on all four completed outputs |
| `person_and_chain_comparisons.csv` | Person/non-person effects and demonstration-chain occurrence |
| `error_categories.csv`, `error_distances.csv` | Taxonomy relationships and measurable error distance |
| `retry_summary.csv`, `token_summary.csv` | Observed attempts and recorded token usage |
| `qualitative_claim_review.csv` | Replay of explicit AI-assisted judgments; not an automatic hallucination rate |
| `regression_checks.json`, `reproduction_manifest.json` | Numerical verification and input/code/environment identity |
| `figures/manifest.json` | Figure-to-CSV and cohort mapping |

The original late-revision analysis is preserved byte-for-byte under `archive/recovered_analysis/revision_20261013/`. Its five core reference tables are regression targets. Later appendix analyses whose unsaved original scripts were unavailable have been reconstructed from manuscript definitions and verified against all 1,698 numeric cells in tables A.1–A.6, including bootstrap endpoints. Their provenance is recorded in `extension_provenance.json`; `manuscript_comparison.csv` provides the cell-by-cell checks. Do not mistake the earlier supplement's fixed-denominator analysis for the final matched-cohort policy.

## Actual final experiment scope

| Run | Scheduled main | Completed main | Failed main | Child-removal records |
|---|---:|---:|---:|---:|
| Gemini | 9,000 | 9,000 | 0 | — |
| Llama seed 0 | 9,000 | 8,991 | 9 | 1,353: 1,352 completed, 1 failed |
| Llama seed 42 | 9,000 | 8,971 | 29 | — |
| Llama seed 123 | 9,000 | 8,986 | 14 | — |

There are **37,353 scheduled records, 37,300 completed and 53 unsuccessful**. Main prompts cover 1,000 entities and nine conditions. The child intervention covers 451 eligible entities and three graph conditions for seed 0. All prompts contain the external Rod Fergusson demonstration (`wd:Q18720219`).

`configs/final_study.json` is the current scope and analysis specification. Root `run_plan.json` is the preserved original proposal, which planned child runs for all three seeds; its planned 40,059 responses are not the observed final study.

## Validate, reconstruct prompts, and test

```bash
python scripts/reproduce_final.py --validate-only
python scripts/prepare_final_prompts.py --output rebuilt-prompts
python -m unittest discover -s tests -v
```

The prompt builder reconstructs the v5-to-v6 migration and compares all generated prompts, metadata and manifests byte-for-byte with the used experiment. It refuses to overwrite an existing output directory.

## Inference and result packaging

Offline replay is the normal reproduction path. To run generation on compatible GPU infrastructure, use the recorded environment in `requirements/final-vllm.txt`, obtain the model through your own authorized access, and read [the execution guide](docs/REPRODUCIBILITY.md). New inference may incur compute/API costs and may not reproduce identical outputs.

```bash
# RunPod, inside the final vLLM environment; optional arguments select seeds.
bash cluster/run_runpod.sh 0 42 123

# Package one observed run, retaining and reporting unsuccessful outcomes.
python scripts/package_results.py --run 'Llama 42' --output llama_seed_42_results.zip
```

New inference is written under a timestamped `results/generated/` directory. Set `RUN_OUTPUT_ROOT` to resume a specific new run. The RunPod wrapper accepts the runner's explicit incomplete-result exit code 2 while preserving its fixed seed. Other errors stop the sequence. It schedules child removal only for seed 0. `launch_llama_three_seeds.sh` is the original CoMa/Apptainer launcher and remains a historical source snapshot.

## Repository map

- `src/graph_typing/`: maintained offline metrics, audits, comparisons, plots and replay.
- `scripts/reproduce_final.py`, `prepare_final_prompts.py`, `package_results.py`: supported final-study commands.
- `configs/`: final scope, raw archive identities and ordered per-file evidence hashes.
- `prompts_v6/`: immutable final prompts, with original paths retained for hashes.
- `results/external_demo/`: final raw responses, attempts and runtime metadata.
- `reports/final_study/`: compact verified final tables, checks and figures.
- `resources/`: compressed full label inventory and checksums; the uncompressed original is ignored.
- `requirements/`: separate analysis, final inference and manuscript environments.
- `manuscript/`: recovered LaTeX source, unchanged empirical text.
- `docs/`: protocol, replay instructions and recovery provenance.
- `third_party/wikc/`: upstream attribution, license, provenance and dataset-construction notebook.
- `archive/`: historical experiments and recovered versioned scripts. Older root generators, v4/v5 prompts, `results/json_mode_final/`, and `reports/final_comparison/` are historical; their names predate the final external-demo study.

The maintained replay package has no runtime import dependency on archived Python code. Existing frozen paths remain intact to preserve provenance. See [recovery notes](docs/RECOVERY.md) for the missing material restored in this repair and the intentionally preserved historical files.

See [validation results](docs/VALIDATION.md) for the repair checks and automated offline workflow.
