# October 2026 final experiment update

- Preserved original main prompts and model outputs.
- Added standalone pinned BF16 greedy Llama runner with validated resume and verification.
- Generated 451 × 3 matched no-child prompts, preserving demonstrations.
- Added flat/micro/hierarchical F1, strict accuracy, distance/directional errors, consistency, paired bootstrap and review worksheets.
- Replaced misleading MRR survival/relative-depth plots; added aligned model and child comparisons.
- Re-scored all 9,000 saved Gemini outputs; added metric definitions, cluster guide and tests.
- LCA-F1 is not implemented; hierarchy metric is explicitly ancestor-augmented F1.

---

# Revision log: 23 September 2026

## Completed

- Placed original research files under `archive/original/` with source, data, prompts and results still together.
- Added project navigation and bibliographic records.
- Replaced the combined/radial presentation with separate score figures, explicit context-source comparisons, output-length dots, and paginated rank/recall heatmaps.
- Recomputed MRR, full-output nDCG, and coverage for all 200 main-run raw results; checked against historical score tables.
- Added paired bootstrap differences and common-cutoff metrics (k = 1, 3, 5, 10), with output padding counted as misses for precision@k.
- Added all missing-type transitions without imputing a rank to absent predictions.
- Added child-availability sensitivity analyses, preserving the distinction from child-only ablation.
- Produced 260 compact review prompts with demonstration/evidence preservation checks and hashes.
- Prepared a deterministic 100-entity child-present cohort with 900 prompts and recorded neighbour IDs/counts.
- Corrected bibliographic attribution and publication-year errors; added relevant 2025 and 2026 papers.

## Not represented as completed

No new inference, prompt-equivalence experiment, raw-Wikidata comparison, ancestor-only condition, matched-token control, or cross-task experiment. The separate-backend relation-only results remain descriptive.
