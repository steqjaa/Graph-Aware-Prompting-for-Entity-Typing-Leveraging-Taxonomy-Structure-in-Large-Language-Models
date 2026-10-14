# Recovery and provenance

Repair base: `d3b9652690755618f5fc53a8d95d2fc0a02865fb`.

The repair restores all final Llama response records and 12 original source files whose comments/docstrings had changed their hashes. Original `frozen_inputs.json` was preserved. The full missing label inventory is bundled as a lossless gzip, below GitHub's per-file limit. The temporary partial label file and large descriptions file are not required by final offline replay and were not reintroduced.

Nine missing scripts are preserved with original text under `archive/recovered_analysis/`:

- `revision_20261013/analysis/`: `recompute.py`, `audit_extra.py`, `build_revision.py`, `finalize_pdf.py`.
- `supplement_20261013/analysis/`: `analyze.py`, `diagnostics.py`, `qualitative.py`, `figures.py`.
- `review_20261011/`: `audit.py`.

`archive/recovered_analysis/provenance.json` records package names, relative paths and SHA-256 hashes. The revision and supplement use different score and bootstrap policies; only the later revision core is the final replay regression target. Historical documentation is retained under `archive/project_documentation_20261012/` and must not be read as current run status.

Newly reconstructed code: v6 migration builder, portable RunPod orchestration, per-run packaging, and later appendix diagnostics. These are labeled reconstructed because exact unsaved terminal sources were unavailable. Core metric semantics and the earlier analysis operation order are retained from recoverable sources. The final manuscript resampling order has additionally been reconstructed and verified: per-comparison RNG reset and original response order, with Gemini order for model contrasts. All 1,698 numeric appendix values match without changing manuscript text. The maintained package extracts shared metrics and the historical response parser so it does not import active logic from `archive/`.

The new layout is deliberately additive. Moving immutable files would invalidate embedded paths, result hashes and resume identities. Older generators/results remain available for development history but are not selected by the final workflow. A future physical cleanup should preserve a complete original snapshot and use an explicit migration manifest.

Upstream WiKC source attribution and license were recovered in `third_party/wikc/`. The upstream license applies to that material; this repair does not unilaterally choose a new license for the thesis author's original work.
