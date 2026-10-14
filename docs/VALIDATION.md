# Repair validation

Validated on the recovered final-study evidence with the pinned CPU analysis dependencies.

- All 21 unit/integration tests pass. Runner tests use mocked generation; no model/API calls are made.
- All 10,353 v6 prompts, 1,451 metadata files, both manifests and both design files reconstruct byte-for-byte.
- The original frozen-input validator passes without changing its expected hashes.
- All 37,353 scheduled final responses pass input identity and evidence checks: 37,300 completed, 53 unsuccessful.
- Five recovered full-precision analysis tables match at absolute tolerance 1e-12.
- All 1,698 numerical appendix values match the unchanged manuscript, including 144 bootstrap endpoints.
- Per-run packaging was exercised for Llama seed 42 and retains its 29 unsuccessful outcomes.
- Generated metric grids and comparison figures were inspected for legibility and clipping.

`reports/final_study/` contains a compact verified output snapshot; regenerate the full analysis with `python scripts/reproduce_final.py`. The reproduction manifest records exact code, evidence and dependency identities. `.github/workflows/reproduce.yml` runs tests, prompt reconstruction and the complete offline replay on pushes and pull requests, and uploads derived artifacts. The workflow has been added; hosted CI execution is separate from the local validation reported here.

The repair does not claim a new GPU inference run or a new external factual review of qualitative judgments. Original archived files retain original whitespace as part of their byte-level provenance.
