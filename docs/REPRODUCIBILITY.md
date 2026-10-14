# Final-study execution and reproduction

## Evidence and policies

The canonical experiment is `configs/final_study.json`. Final raw outputs are under `results/external_demo/`; historical output directories are never pooled into replay. `raw_files.jsonl` records ordered files and hashes from the four supplied ZIPs. Original archive hashes are in `raw_archives.json`. The order is preserved because the recovered core analysis uses a shared bootstrap RNG.

Primary condition means include completed responses. Each paired comparison uses the intersection of completed outputs required by that contrast. Three-seed summaries use a common completed cohort per condition. Failures remain visible, with scheduled/completed/failed counts and separate zero-utility/bounded sensitivity. Bootstrap intervals use 2,000 entity resamples and seed 230917. They condition on saved outputs, not independent draws of model seeds.

The final manuscript calculations reset their RNG for each comparison and retain original raw-response entity order (Gemini order for model comparisons). This reconstructed ordering reproduces every printed appendix bootstrap endpoint. The recovered core retains the original raw order and shared-RNG operation order to reproduce the saved tables exactly.

## Commands

`python scripts/reproduce_final.py` performs resource restoration, frozen validation, raw evidence validation, scoring, audits, reference-table checks, appendix diagnostics, qualitative review and plots. `--validate-only` checks evidence without calculating scores; `--no-plots` retains numerical analysis and skips figure rendering. `--output PATH` writes derived artifacts to a separate directory.

The package reads the original full label inventory but only materializes label candidates queried by the recorded predictions. It scans every source row, retaining all candidates including IDs outside the graph. This preserves ambiguity semantics while using less memory than loading millions of irrelevant labels. It is not the graph-only subset used in an earlier supplement.

`python scripts/restore_resources.py` restores the missing 268,048,010-byte label inventory from its exact compressed representation. The full SHA-256 remains `4e557be512f94e96fc27462906aa55ea22857b514e27600418cf5c8d304e9c72`; historical frozen hashes were not rewritten.

## Inference

The supplied final archives record vLLM 0.31.0, Torch 2.13.0, Transformers 5.17.0, CUDA 13.0, and NVIDIA H100 80GB. The Llama checkpoint revision is `0e9e39f249a16976918f6564b8830bc894c89659`. The original source and parser snapshots have been restored, so resume identities match the recorded configurations.

The supplied raw directories are the immutable thesis evidence. Back them up and use a separate checkout/output root for new experiments. The old inference runner supports `--output-root`; its resume logic preserves successful outputs but may replace unsuccessful ones while archiving prior attempts. Do not treat rerunning inference into the thesis evidence as offline reproduction.

The portable wrapper writes new inference under `results/generated/<UTC timestamp>/`; set `RUN_OUTPUT_ROOT` to reuse a new run directory. `cluster/run_runpod.sh` follows the recorded final scope and checks pipeline exit status even when using `tee`. Code 2 means saved unsuccessful outcomes; it is reported and the next seed proceeds. An exception or other error stops. The original CoMa launcher is retained only as historical code.

Gemini generation remains `run_gemini_external.py`; an API key is taken from `GEMINI_API_KEY` or a hidden prompt. No credentials are included in this project. Model/API availability and provider behavior are outside offline replay guarantees.

## Packaging

Use `scripts/package_results.py` to create per-run or combined archives. It checks scheduled coverage, entity/prompt/metadata identity, hashes, model/seed, checkpoint and accepted-response parsing. It includes unsuccessful records, attempts, runtime/config files and available logs. `package_manifest.json` reports completion and hashes every included source file. `--require-success` is an optional stricter gate.

Packaging does not make missing tasks acceptable: an absent scheduled response or unexpected result is rejected. An unsuccessful saved response is an observed experimental outcome and is preserved.

## Qualitative review

`qualitative.py` replays the original recorded judgment rules and source URLs for 108 outputs. Supported additions are reviewer interpretations. Gold agreement is not external factual certification, and outside-inventory labels are not automatically hallucinations. These judgments must be assessed as AI-assisted qualitative annotations, not an independently annotated population estimate.

## Manuscript and plots

The LaTeX source is recovered under `manuscript/`. The repair does not silently rewrite its empirical text. Regenerated plots use explicit numerical labels and CSV mappings. Recovered revision core tables are checked at absolute tolerance 1e-12; manuscript tables round those values. All 1,698 numeric values in appendix tables A.1–A.6, including reconstructed extensions and all bootstrap endpoints, are verified against the unchanged manuscript. A mismatch fails the replay and is recorded in `manuscript_comparison.csv`.

To build the existing LaTeX document, use its own README and `latexmkrc`. Historical ReportLab/PyMuPDF builders are archived separately and have different dependencies. Analysis replay does not require either document toolchain.
