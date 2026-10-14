# Historical vLLM greedy workflow

This guide records an earlier backend configuration; the current rerun entry points are documented in README_RERUNS.md. The original Transformers runner is retained
unchanged for reproducing the previous backend. The fast runner uses vLLM,
BF16, the same pinned Llama 3.1 8B revision, greedy decoding and unchanged prompts.
Use a separate results directory for each backend; do not merge their responses.

## Work already completed

- Six source archives were combined, preserving main prompts and results.
- Gemini fresh: 9,000 naturally stopped, valid responses, re-scored and plotted.
- Main Llama workload: 9,000 responses. Matched children-removed: 1,353 responses
  for 451 entities, comparing against the full-context arm of the main Llama run.
- The Archive 5 add-on was inspected but not applied: its runner can mark token-
  limited responses completed and weakens resume provenance checks. The stricter
  original runner/scorer and plot suite have been preserved and extended.

## Connection and allocation

Use the established CoMa connection and GPU allocation workflow. The historical account was `sindi.teqja@cluster.ginkgo-project.de`. Run inference on an allocated GPU compute node.

From your Mac, in this extracted project directory:

```bash
ssh sindi.teqja@cluster.ginkgo-project.de 'mkdir -p ~/thesis-final'
rsync -az --progress scripts/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/scripts/
rsync -az --progress cluster/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/cluster/
rsync -az --progress prompts_v5/all_1k/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/prompts_v5/all_1k/
rsync -az --progress prompts_v5/children_removed/ sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/prompts_v5/children_removed/
rsync -az --progress requirements-llama-fast.txt sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/
```

## Fast backend setup (once, in an approved GPU environment)

If your node permits Python environments with a supported CUDA driver:

```bash
cd ~/thesis-final
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements-llama-fast.txt
python -m pip freeze > vllm-environment-lock.txt
```

vLLM has tightly coupled CUDA/PyTorch dependencies: use a fresh environment.
If COMA requires Apptainer, use a cluster-approved vLLM/CUDA image with `--nv`
and invoke the same runner inside it; the old `/tmp/pytorch.sif` image is not
assumed to contain vLLM. Do not use the old `/tmp/hf-python` PYTHONPATH with this
fresh environment. Reuse your established Hugging Face cache/login; model access
must already be approved for the pinned gated Meta model.

## Validate, smoke test, then run both experiments

```bash
python scripts/run_llama_fast.py --dry-run
bash cluster/run_fast.sh --max-jobs 9
bash cluster/run_fast.sh
python scripts/run_llama_fast.py --verify-only
```

The smoke test runs nine pending jobs PER experiment, including reasoning methods.
It returns exit 2 while the full workload is missing; inspect the saved JSON and
log. The subsequent full command reuses valid smoke results, loads the model once,
runs both manifests, and automatically retries non-completed results up to twice
per invocation. A full run exits 2 if ANY failed/missing responses remain. Rerun the
same command to resume only unfinished jobs; completed records are never replaced.

Default output folders are `results/llama_1k_fast` and
`results/llama_children_removed_fast`.

Final verification must show 9,000 main completions and 1,353 child removals,
with zero failed and zero missing in both `completion.json` files.

## Tokens and throughput

Default output allowance: **16,384 new tokens**, with early natural stop. A
token-limited response is incomplete even if its JSON looks valid. On the second
attempt the allowance doubles, bounded by the remaining configured context.
No inputs are truncated. Stop reasons, actual token counts, attempts, runtime
versions and prompt/model hashes are saved.

For no extra output ceiling beyond the configured context:

```bash
bash cluster/run_fast.sh --max-new-tokens 0
```

Choose that setting BEFORE starting, or use a new output root. The default total
context is 32,768 tokens (input plus output). The model's context window, available
KV-cache memory and a natural stop still impose limits. A truly unlimited run is
not possible; pathological non-stopping generations remain failures for review.
Raising the ceiling does not force a normal answer to use all those tokens.

The engine continuously batches within chunks of 128 jobs, with up to 32 active
sequences and shared-prefix caching. There is **no artificial RPM cap**. Throughput
is measured as valid completed responses/minute in `throughput.json`, excluding
model load; disk writes and retries are included. Actual speed must be measured on
your GPU. At a measured 150/minute, 10,353 responses take about 69 minutes of
inference, plus allocation, installation/model loading, retries and evaluation.
This is arithmetic, not a runtime promise.

Benchmark a larger concurrency setting in a separate scratch output folder:

```bash
bash cluster/run_fast.sh --max-jobs 128 --max-num-seqs 64 --output-root results/benchmark64
```

Choose the settings that produce the highest VALID completion rate, then keep
them fixed for the entire final experiment. All configuration changes require a
new output root. Do not silently mix benchmark results with the final run. If the
node has multiple GPUs, independent replicas may outperform tensor parallelism
for an 8B model; this package does not automatically reserve additional resources.

Repeated malformed greedy output may repeat on retry. The runner preserves it as
failed and retains every previous attempt. It does not alter the prompt, sample
until an answer passes, constrain JSON generation, or fabricate a completion.

## Copy back and produce all final evaluations

From the full local project folder, after GPU verification succeeds:

```bash
rsync -az --progress sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/results/llama_1k_fast/ results/llama_1k_fast/
rsync -az --progress sindi.teqja@cluster.ginkgo-project.de:~/thesis-final/results/llama_children_removed_fast/ results/llama_children_removed_fast/
python3 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/evaluate_all.py
```

This re-scores the three final runs with identical metrics and produces:

- Macro/Micro-F1, precision/recall, ancestor-augmented hierarchical F1, exact-set
  accuracy, most-specific hits, MRR/nDCG, coverage at fixed ranks.
- Taxonomy vs no taxonomy for direct, CoT and contrastive prompting, with paired
  entity confidence intervals, context-component effects and depth analysis.
- Llama vs Gemini per condition, paired F1 differences and difference in taxonomy
  uplift between model systems.
- Matched with-children vs without-children effects on the same 451 entities.
- Error distance/direction, hierarchy consistency, representative helped/harmed
  cases and human-review worksheets.
- 15 PNG/SVG figure pairs under `results/llama_1k_fast/analysis/story_figures/`;
  numeric companions and a report under `reports/final_comparison/`.

The supplied Gemini-only evaluation/report can be reproduced immediately:

```bash
python scripts/evaluate_all.py --gemini-only
python scripts/comparison_report.py --output reports/gemini_available
```

The older partial Gemini run and historical Llama pilot remain separate. They do
not supply missing final-run outcomes. The current child-removal experiment is
planned for Llama; a Gemini child-removal arm was not part of that run and
cannot be claimed. This comparison evaluates model SYSTEMS: Gemini temperature
1/provider thinking versus greedy BF16 Llama/vLLM, using oracle taxonomy context.
