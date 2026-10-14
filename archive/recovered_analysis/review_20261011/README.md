# Thesis review analysis supplement

This package supports the revised thesis and its comment-by-comment review response. It contains reanalysis of existing archived outputs, not new model inference.

## Reproduce

Requires Python 3, pandas and numpy. Extract the supplied Thesis_final_complete_project.zip so its thesis-final directory is available, then run:

    python audit.py --project /path/to/thesis-final --output audit-output

Results are written beneath audit-output/work. The source archive is required and is not duplicated here. The nine exact example_0002 prompts and their Standard/graph-aware difference are included separately.

## Files

- entity_audit.csv: graph context availability, truncation, gold-set closure, depth ties and literal centre-label presence for 1,000 entities.
- stats.json: aggregate graph and cohort statistics.
- retries.csv/json: first-call validity, recovered jobs and observed attempts by condition, deduplicating archived records of the same call.
- thinking.csv/json: Gemini reported thinking-token totals and distribution summaries.
- lengths.csv/json: generated type counts.
- contrasts.csv/json: paired F1 contrasts for label-presence groups and the 549 empty-child cases. Percentile intervals use 2,000 entity bootstrap samples, seed 230917.
- *_prompt.txt and prompt_diff.txt: unmodified archived prompt texts and their textual comparison.

## Interpretation

Literal presence uses casefolding, underscore/space normalization and boundary punctuation stripping, followed by substring matching; it is not semantic leakage detection. Empty-child comparisons retain differences in instructions and formatting, so they do not isolate an instruction-only causal effect. Retry summaries describe archived calls, not unlogged calls. The audit uses the final fresh Gemini run, final Llama main run, and final Llama children-removed run. It does not pool exploratory runs.

The revised manuscript separately reports graph-absent IDs, seed exceptions, hardware, run dates and exact retry policies, verified against project metadata and scripts. Two reviewer requests remain open: an external demonstration followed by rerunning inference, and three independent seeds. See Review_response.docx for all 45 comments and their disposition.
