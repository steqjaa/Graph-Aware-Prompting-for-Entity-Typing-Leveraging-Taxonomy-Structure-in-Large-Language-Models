# Thesis project

Complete experiment project with historical data, source code, prompts, raw model outputs, attempts, archives, reports and figures. New external-demonstration prompts are in `prompts_v6/`; historical prompts remain in `prompts_v4/`, `prompts_v5/` and `archive/`.

## New experiments

The demonstration is Rod Fergusson (`wd:Q18720219`), a WiKC entity outside the 1,000-entity evaluation cohort. The four planned runs are one Gemini run and three Llama runs with seeds 0, 42 and 123. Temperature remains 1.0. The runs have not yet been executed.

On the Gemini machine:

```bash
bash launch_gemini.sh
```

On the allocated CoMa GPU machine:

```bash
bash launch_llama_three_seeds.sh
```

The Llama launcher runs the main and matched child-removal conditions for each seed. New outputs are saved under `results/external_demo/`. Existing results are not overwritten. See `README_RERUNS.md` for environment requirements, resume and verification commands.

## Historical final results

| Experiment | Directory | Accepted responses |
|---|---|---:|
| Gemini main | `results/gemini_1k_fresh/` | 9,000 |
| Llama main | `results/json_mode_final/llama_1k_fast/` | 9,000 |
| Llama child removal | `results/json_mode_final/llama_children_removed_fast/` | 1,353 |

Earlier runs, raw outputs, attempts and submitted code remain in `results/` and `archive/`. Historical figures and reports remain in `figures/`, `analysis/` and `reports/`.

Historical generation settings and evaluation commands are documented in `docs/FINAL_RUN_AND_EVALUATION.md`; metric definitions are in `docs/EVALUATION_METRICS.md`. Guides for earlier backends are labelled historical. Bibliographic records remain in `docs/references.md` and `docs/references.bib`.
