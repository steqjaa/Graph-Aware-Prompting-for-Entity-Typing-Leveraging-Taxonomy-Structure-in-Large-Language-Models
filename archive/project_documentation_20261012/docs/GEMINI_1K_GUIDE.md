# Historical Gemini v5 workflow: 1,000 entities, nine conditions

This add-on supplies 9,000 ready-to-run compact prompts. It excludes retrieved and shuffled conditions from the new run. Historical files in archive/original are untouched.

## Setup (macOS/Linux terminal)

Extract thesis-gemini-complete.zip, then open Terminal in the included thesis-cleaned folder. No add-on is needed.

```bash
cd /path/to/thesis-cleaned
python3 -m venv .venv
source .venv/bin/activate
python -m pip install networkx pandas numpy matplotlib
```

Inference itself uses Python's standard-library REST client, not a Gemini SDK. No local GPU is needed for inference. The API key is supplied through the environment or terminal prompt.

## Check prompts without API access

```bash
python scripts/run_gemini.py --dry-run
```

Expected: Entities 1000, conditions 9, total 9000. There are 1,000 original dataset IDs, each paired with:

1. standard
2. graph_aware_standard
3. cot_no_graph
4. cot_graph
5. contrastive_cot_no_graph
6. contrastive_cot_graph
7. standard_parent_only
8. standard_siblings_only
9. standard_children_only

## Model and credentials

The API key is requested with hidden input whenever GEMINI_API_KEY is absent. For the simplest workflow, just paste it at each hidden prompt. The runner never writes it to output files.

```bash
python scripts/run_gemini.py --list-models
```

Choose a text model that your project can access. The default is gemini-3.8-flash, listed in Google's model documentation when checked on 24 September 2026. You can pass `--model YOUR_MODEL_ID` instead. Keep the model and generation settings fixed throughout a run. Avoid auto-updating `latest` aliases for research; returned modelVersion is saved and must still be checked for changes.

## Smoke test

```bash
python scripts/run_gemini.py --max-calls 9 --workers 1 --rpm 10
```

This runs one entity in nine conditions. Inspect the result files in results/gemini_1k/example_0001/. Each should have status completed, an appropriate JSON response, and predictions. The smoke entity is the original demonstration entity and is deliberately zero-shot; run `--max-calls 9` again to test the next entity's normal worked-example format.

A run uses temperature 1.0, a 4,096 output-token budget, JSON response MIME, and the provider's default thinking setting. This is a new Gemini experiment, not a replication of the earlier Llama decoding configuration. The model's requested explanation is observable output; it is not access to private internal reasoning. Output limits can include model thinking depending on the model; inspect finish_reason and usage rather than assuming all tokens are answer text.

If any model/configuration settings must change, use a new `--output results/NAME` directory and keep the same settings on every subsequent command. Existing results cannot be mixed with different settings or changed inputs.

## Run remaining prompts

```bash
python scripts/run_gemini.py --workers 4 --rpm 30
```

The runner skips completed results and prints progress. `--rpm` controls start rate across workers; it does not enforce token-per-minute/day quotas. Set it below the limit shown for your project/model in AI Studio. Four workers is a configurable default, not a guarantee of optimal throughput. The machine must stay awake and the terminal must remain open. On macOS, prefix the command with `caffeinate -i` to prevent idle sleep during the run.

There are 9,000 initial generation jobs. Transient retries can add API requests and cost. At 30 requests/minute, the rate-only lower bound is 300 minutes (5 hours); actual time can be longer due to latency, token limits, quotas, and retries. Do not assume a free-tier quota can finish this run in a day. Check account limits and usage/pricing before the full run; no cost estimate is hard-coded.

## Resume and retry

After a network interruption or a clean stop, run the same command again. Completed results are reused. Saved failures are skipped by default. To retry them:

```bash
python scripts/run_gemini.py --workers 2 --rpm 10 --retry-failed
```

Transient 429/5xx/network errors receive bounded backoff. Persistent 429 errors stop further jobs after limited attempts. Authentication and configuration HTTP errors stop the queue. Non-STOP finishes, blocked responses, and invalid JSON are saved as failures, not counted as completed empty predictions. Retries retain prior attempt files. An interruption after a request reaches Google but before its result is saved can require reissuing that request; exactly-once billing is not guaranteed.

Only one process may write a result directory. A hard kill can leave results/gemini_1k/RUNNING.lock. Remove that file only after confirming no runner process is still using the directory.

## Score and plot

```bash
python scripts/score_gemini.py
```

For custom paths, pass the same `--manifest` and `--results` paths. Completion status is written even when the run is incomplete; aggregate scores are withheld until all expected jobs have completed successfully. Outputs appear in results/gemini_1k/analysis/:

- completion_status.csv and entity_scores.csv
- summary_all_1000.csv
- summary_few_shot_only.csv (999 entities)
- summary_children_present.csv
- gold_type_ranks.csv (blank rank means missing)
- five PNG/SVG plots: MRR, nDCG, most-specific recovery, coverage, and mean output length
- interpretation.json, including observed model versions

The new plots are simple full-run summaries without bootstrap intervals. The original 17 pilot plots are not silently overwritten. Inspect model versions before pooling results; if versions changed during the run, stratify or rerun under a fixed available version.

## Dataset and design decisions

All 1,000 entities are included, including entities previously inspected. This is exploratory re-evaluation, not a new held-out test. Neighbourless cases have empty context; no entity is dropped. The generator caps each relation at three nodes, so full context can contain up to nine nodes. This is not a matched total-node or token budget.

For entities with several graph-minimal annotated types, choose the deepest one, breaking ties by class ID, as a single oracle centre. All original gold types remain in scoring. The audit lists the alternative centres and chosen one. Do not interpret this as complete neighbourhood evidence for all branches.

The original worked example belongs to the 1k set. Its own test prompts omit every example block to avoid exposing its answer. The other 999 retain the reviewed example. Report the 999-case few-shot summary alongside the 1,000-case summary because demonstration exposure differs. Report child-present and all-entity results separately.

All graph conditions use privileged gold-derived context. They are oracle diagnostics, not retrieval experiments. Historic MRR uses maximum annotated depth; historic nDCG uses depth-normalized gains. Exact normalized-label matching is retained. The scorer uses the archived metric definitions documented in experimental_protocol.md; nDCG's general rationale is Järvelin and Kekäläinen (2002), bibliography [7]. Prompt families remain adaptations of Wei et al. (2022) and Chia et al. (2023), bibliography [5,6]. The new full-cohort rules are design choices, not claims established by those papers.

## Regeneration (optional)

The add-on already contains the prompts. To regenerate into a separate directory:

```bash
python scripts/prepare_all_1k.py --output prompts_v5/all_1k_regenerated
```

Then pass its manifest to the runner/scorer. Do not regenerate or modify inputs in place after starting inference. The runner fingerprints prompts, metadata, manifest, configuration and runner source to prevent mixing incompatible experiments.

## Verification and API references

Checked locally: 1,000 unique entities; 9,000 unique entity/condition pairs; absence of retrieved/shuffled conditions; example removal for the demonstration entity; JSON parsing, blocked/truncated-response handling; completed-result resume; failure retry; API-key non-persistence; scoring and five plots using mock responses. **No live Gemini call was made during development**, since the key was not supplied here. The smoke test is the live compatibility check.

Official documentation checked 24 September 2026:

- GenerateContent REST endpoint and configuration: https://ai.google.dev/api/generate-content
- Model listing: https://ai.google.dev/api/models
- Available model catalogue: https://ai.google.dev/gemini-api/docs/models
- Project rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
