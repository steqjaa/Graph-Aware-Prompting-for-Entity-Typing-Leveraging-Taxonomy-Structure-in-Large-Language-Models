# Thesis revision: analysis supplement

This package contains the numerical evidence, prompt strings, output review, and analysis code supporting the revised thesis. All experiment scores derive from the four newly supplied archives. No previous thesis experiment scores are used.

## Contents

- `output/condition_scores.csv`: primary fixed-denominator results; failed final tasks contribute zero metric scores.
- `output/accepted_scores.csv`: scores conditional on successful outputs.
- `output/entity_scores.csv`: per-task metrics, completion, depth, description stratum, output counts, and token counts.
- `output/paired_comparisons.csv`: paired graph contrasts, model contrasts, empty-child controls, child interventions, and 5,000-replicate percentile intervals. Delta is the first named condition minus the comparator; child deltas are included minus removed.
- `output/seed_summary.csv`: mean, sample standard deviation, minimum, and maximum of three Llama run-level mean F1 scores.
- `output/prediction_audit.csv`: every unique normalized accepted predicted label and its benchmark/taxonomy category. These categories are not factuality judgments.
- `output/coverage_summary.csv`: sums and micro gold coverage by condition; the thesis primarily reports entity-macro metrics.
- `output/qualitative_outputs.json`: actual predictions, explanations, and gold for 108 outputs (3 entities × 9 conditions × 4 runs).
- `output/qualitative_claim_review.csv`: all 954 label occurrences in those outputs, with judgment, rationale, and source URL. Broad entailments are reviewer inferences. This is a purposive AI-assisted case review, not an independently rated population estimate.
- `output/retry_audit.csv`, `retry_summary.csv`, `incomplete_results.json`: observed attempt histories and unresolved tasks. A successful observed history does not reveal an unrecorded configured transport retry cap.
- `output/prompts/`: all nine complete prompts for Sony Alpha 230 and exact child-ablation differences. Top-level prompt diffs use entity 1 to isolate instruction and formatting changes.
- `output/figures/`: publication figures, with exact labels.
- `output/references.json`: research bibliography; the thesis includes conventional citations.
- `output/provenance.json`: input hashes and validation checks.
- `analysis/source/`: WiKC graph, evaluation data, construction notebook, taxonomy-node label subset, upstream attribution and license.

## Recompute

Use the Python version recorded in `ANALYSIS_ENVIRONMENT.json` and the package versions in `requirements-replay.txt` (the unpinned `requirements.txt` is also retained). From this directory, create an `upload` directory and copy in the original, unmodified archives:

- `gemini.zip`
- `llama_seed_0_results.zip`
- `llama_seed_42_results.zip`
- `llama_seed_123_results.zip`

Run these commands in order:

```bash
python analysis/analyze.py
python analysis/diagnostics.py
python analysis/qualitative.py
python analysis/figures.py
```

The first script creates a large intermediate `analysis/records.pkl`. It is omitted from this package because it is reproducible from the supplied archives. The archives themselves are not duplicated here. The semantic labels in `qualitative.py` are explicit review decisions, not automated factuality predictions; rerunning it reproduces the recorded judgments, not new external verification. Minor CSV row-order variation can arise from Python sets without affecting aggregate results.

## Interpretation

There are 36,000 main tasks and 1,353 seed-0 child-ablation tasks. The main study has 52 unresolved final tasks, and the ablation has one. Each main condition contains 1,000 scheduled tasks; ablation conditions contain 451. Accepted-output and complete-pair analyses supplement the fixed denominator. Three seeds describe only the observed Llama runs. Bootstrap intervals resample paired entities with seed 20261013; they are conditional on the recorded outputs and have no multiplicity adjustment.

Normalization lowercases labels, changes underscores to spaces, collapses whitespace, and removes outer ASCII punctuation. The thesis defines flat, ranking, and hierarchical metrics, including unknown/ambiguous label handling. A non-gold label is not automatically a hallucination, and a gold match is not independent evidence of truth. The qualitative review distinguishes supported additions, contradicted categories, malformed output, and unresolved assertions.

The graph treatment also changes instructions and demonstration presentation. The child intervention removes child-specific wording alongside evidence. Neither contrast isolates an evidence-only causal effect.

WiKC source material is credited to Yiwen Peng and collaborators; see `analysis/source/LICENSE`, `README.md`, and `SOURCE_PROVENANCE.json`. Taxonomy labels were reduced to the exact 16,581 graph-node rows to keep this package compact; the analysis only uses those rows. Existing input archive hashes remain in `output/provenance.json`.

## Research integrity revision

Appendices D–F of the updated thesis document replay, retention, attribution, AI assistance and an unsigned receipt of the supplied guidelines. `Integrity_and_archive_handover.md` records actions for the author and supervising unit. Institutional deposit, final author verification, permission for assistance, and signatures are not asserted as completed. Numerical tables and empirical findings are unchanged.
