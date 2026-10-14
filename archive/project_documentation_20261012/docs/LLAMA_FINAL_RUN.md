# Historical Llama backend workflow

Updated 9 October 2026. This guide supersedes older instructions about headline metrics and the final run. The main prompts must not be regenerated.

## Workload and files

- Main run: 1,000 entities × nine conditions = **9,000 calls**.
- Matched child removal: 451 child-present entities × three graph methods = **1,353 calls**.
- Total: **10,353**, reusing the full-graph main results as the children-included arm.
- `prompts_v5/children_removed/` is already generated in this package. Its manifest has 1,354 lines including the header.
- New conditions are `graph_aware_standard_no_children`, `cot_graph_no_children`, and `contrastive_cot_graph_no_children`. `source_condition` links each to the main run.
- Test-task child context and child-specific preamble instructions are removed. Parents, siblings, target description, output schema and methodology are retained. The entire demonstration is preserved verbatim, including its empty child context and explanation. This isolates removal of test-task child evidence; it does not remove every use of the word “child” from the demonstration.
- Metadata `children_available` records original cohort availability; `children_removed` indicates the intervention. `source_context_ids` preserves the original context; `context_ids.children` is empty.

The generator refuses to overwrite an existing output directory. To reproduce it for an audit:

```bash
python3 scripts/prepare_children_ablation.py --output prompts_v5/children_removed_audit
```

Use the provided main and child manifests for inference. An audit output has different paths and is not a substitute after inference begins.

## Transfer only the runner and prompts to CoMa

From the project root on your Mac:

```bash
ssh sindi.teqja@cluster.ginkgo-project.de 'mkdir -p ~/thesis-cleaned/scripts ~/thesis-cleaned/prompts_v5'
rsync -az --progress scripts/run_llama.py sindi.teqja@cluster.ginkgo-project.de:~/thesis-cleaned/scripts/
rsync -az --progress prompts_v5/all_1k/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-cleaned/prompts_v5/all_1k/
rsync -az --progress prompts_v5/children_removed/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-cleaned/prompts_v5/children_removed/
```

The runner is standalone. It does not need the dataset, scoring modules, Gemini results or archived environment on CoMa. The container must have CUDA, PyTorch and Transformers installed, and the account must already have access to the gated Hugging Face Llama model. `requirements-llama.txt` documents compatible package ranges; keep the actual environment fixed throughout a run.

## Inference on your allocated H100 node

```bash
cd ~/thesis-cleaned
apptainer exec --nv \
  --env PYTHONPATH=/tmp/hf-python \
  --env HF_HOME="$HOME/hf_cache" \
  /tmp/pytorch.sif \
  python scripts/run_llama.py \
    --manifest prompts_v5/all_1k/manifest.csv \
    --output results/llama_1k --batch-size 8 --max-jobs 9
```

Inspect the nine result files and statuses. If any fail, investigate the saved response before continuing. Then run the complete main manifest in a persistent terminal session or your cluster's normal scheduled-job workflow:

```bash
set -o pipefail
apptainer exec --nv \
  --env PYTHONPATH=/tmp/hf-python \
  --env HF_HOME="$HOME/hf_cache" \
  /tmp/pytorch.sif \
  python -u scripts/run_llama.py \
    --manifest prompts_v5/all_1k/manifest.csv \
    --output results/llama_1k --batch-size 8 \
  2>&1 | tee ~/llama_1k.log
```

Smoke-test the child prompts (three calls), then repeat without `--max-jobs 3`:

```bash
apptainer exec --nv \
  --env PYTHONPATH=/tmp/hf-python \
  --env HF_HOME="$HOME/hf_cache" \
  /tmp/pytorch.sif \
  python -u scripts/run_llama.py \
    --manifest prompts_v5/children_removed/manifest.csv \
    --output results/llama_children_removed --batch-size 8 --max-jobs 3
```

Pinned inference setup:

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Revision | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Precision | BF16 |
| Decoding | Greedy, no sampling, one beam |
| Maximum new tokens | 800 |
| Input | One user message, model chat template, left padding |
| Stop | EOS or end-of-turn token |

Completed results are reused; prompt, metadata, manifest, model, runner, batch-size and runtime changes are rejected. Batch size is recorded because numerical differences can affect greedy decoding. Missing jobs resume automatically. Saved failures require `--retry-failed`, which preserves the previous result under `attempts/`. A repeated greedy malformed response may fail again; do not change generation settings inside an existing experiment directory.

Atomic writes protect completed JSON files. OS file locking releases automatically on process exit. A crashed batch may be rerun; no completed file is lost. Prompts are never truncated. Outputs that exhaust 800 tokens without a stop token remain `incomplete` even if their text looks parseable.

## Verify without a GPU or model download

```bash
python3 scripts/run_llama.py --manifest prompts_v5/all_1k/manifest.csv --output results/llama_1k --verify-only
python3 scripts/run_llama.py --manifest prompts_v5/children_removed/manifest.csv --output results/llama_children_removed --verify-only
```

Expected:

```text
Completed 9000/9000 | failed 0 | missing 0
Completed 1353/1353 | failed 0 | missing 0
```

Verification returns exit code 2 if incomplete. A dry run is also available via `--dry-run` and makes no model calls.

## Copy results back and score

From the local project root:

```bash
rsync -az --progress sindi.teqja@cluster.ginkgo-project.de:~/thesis-cleaned/results/llama_1k/ results/llama_1k/
rsync -az --progress sindi.teqja@cluster.ginkgo-project.de:~/thesis-cleaned/results/llama_children_removed/ results/llama_children_removed/
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/score_gemini.py --manifest prompts_v5/all_1k/manifest.csv --results results/llama_1k
python scripts/score_gemini.py --manifest prompts_v5/children_removed/manifest.csv --results results/llama_children_removed
python scripts/score_gemini.py --manifest prompts_v5/all_1k/manifest.csv --results results/gemini_1k_fresh
python plot_stories.py \
  --results results/llama_1k --primary-label 'Llama 3.1 8B Instruct' \
  --compare-results results/gemini_1k_fresh --compare-label 'Gemini 3.8 Flash' \
  --children-results results/llama_children_removed
```

`score_gemini.py` retains its filename for compatibility, but scores either model with identical functions. It refuses missing, failed, mismatched or unexpected results. Existing human annotations are never overwritten. `summary.csv` is the current summary; `summary_all_1000.csv` remains a compatibility name (its `n` column is authoritative even for a 451-entity child run).

Without Llama results, regenerate the included Gemini-only figures with:

```bash
python plot_stories.py --results results/gemini_1k_fresh --primary-label 'Gemini 3.8 Flash'
```

Plots live under `<results>/analysis/story_figures/`. The complete run creates 15 PNG/SVG figure pairs, including F1, hierarchical F1, gold coverage, supporting nDCG and rank buckets, context ablations, depth uplift, paired graph effects, model comparisons, matched child effects, error distance/direction and consistency. Micro-F1, exact-set accuracy, MRR and all numeric metrics are in `summary.csv`. CSV companions include confidence limits and support counts. The old MRR survival and relative-depth “specificity” plots are removed from the current figure directory.

## Interpretation and human review

See `EVALUATION_METRICS.md` for exact definitions. Gemini uses its recorded provider configuration (temperature 1.0 and provider-default thinking); Llama uses BF16 greedy decoding. Describe the comparison as a **model-system comparison**, not evidence that architecture alone caused a difference. Graph context is oracle context; previously examined entities remain included. The 999 few-shot cases are also summarized separately.

`manual_validation_sample.csv` samples up to 20 incorrect predictions per condition/category. Fill in validity, evidence, annotator and notes; review against the description and independent evidence. “Outside WiKC” does not mean hallucinated. Sampling is stratified and exploratory, so raw sample percentages are not unbiased overall error rates. `representative_cases.csv` contains helped/harmed examples with gold and both prediction sets. No human judgments have been fabricated.

Run local tests with `python -m unittest discover -s tests -v`. No tests make model calls. The prepared package includes a complete re-score of the existing 9,000 Gemini outputs; Llama inference itself was not executed during this update.
