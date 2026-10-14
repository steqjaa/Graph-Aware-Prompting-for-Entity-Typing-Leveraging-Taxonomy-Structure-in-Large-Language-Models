# Final experiment evaluation

The primary research contrast is graph-aware versus no-graph prompting within each model. Model comparisons and child ablations are supporting analyses.

Scores below use completed, matched main-run results. Taxonomy context is oracle context.

| Model system | Condition | n | Macro-F1 | Micro-F1 | Hierarchical F1 | Most-specific hit |
|---|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | Contrastive + taxonomy | 1000 | 0.4739 | 0.4619 | 0.8349 | 0.6390 |
| Gemini 3.8 Flash | Contrastive | 1000 | 0.3376 | 0.3297 | 0.6866 | 0.3130 |
| Gemini 3.8 Flash | CoT + taxonomy | 1000 | 0.4735 | 0.4643 | 0.8332 | 0.6330 |
| Gemini 3.8 Flash | CoT | 1000 | 0.3322 | 0.3248 | 0.6693 | 0.2930 |
| Gemini 3.8 Flash | Standard + taxonomy | 1000 | 0.4582 | 0.4515 | 0.8001 | 0.6150 |
| Gemini 3.8 Flash | Standard | 1000 | 0.3876 | 0.3816 | 0.6994 | 0.4450 |
| Gemini 3.8 Flash | Children only | 1000 | 0.4238 | 0.4163 | 0.7284 | 0.4930 |
| Gemini 3.8 Flash | Parent only | 1000 | 0.4271 | 0.4213 | 0.7603 | 0.5720 |
| Gemini 3.8 Flash | Sibling only | 1000 | 0.3894 | 0.3826 | 0.7354 | 0.5700 |
| Llama 3.1 8B Instruct (vLLM) | Contrastive + taxonomy | 1000 | 0.3721 | 0.3610 | 0.7252 | 0.3150 |
| Llama 3.1 8B Instruct (vLLM) | Contrastive | 1000 | 0.3123 | 0.3032 | 0.5673 | 0.2420 |
| Llama 3.1 8B Instruct (vLLM) | CoT + taxonomy | 1000 | 0.3829 | 0.3736 | 0.7080 | 0.3040 |
| Llama 3.1 8B Instruct (vLLM) | CoT | 1000 | 0.3209 | 0.3126 | 0.5590 | 0.2080 |
| Llama 3.1 8B Instruct (vLLM) | Standard + taxonomy | 1000 | 0.4059 | 0.3888 | 0.7211 | 0.2530 |
| Llama 3.1 8B Instruct (vLLM) | Standard | 1000 | 0.3360 | 0.3221 | 0.5601 | 0.1930 |
| Llama 3.1 8B Instruct (vLLM) | Children only | 1000 | 0.3499 | 0.3354 | 0.5775 | 0.1890 |
| Llama 3.1 8B Instruct (vLLM) | Parent only | 1000 | 0.3953 | 0.3793 | 0.7114 | 0.2350 |
| Llama 3.1 8B Instruct (vLLM) | Sibling only | 1000 | 0.3393 | 0.3280 | 0.5941 | 0.2010 |

Hierarchical F1 uses ancestor closure, including self and roots; it is not LCA-F1.
Gemini and Llama use different inference systems and decoding settings. The comparison does not isolate architecture.
Confidence intervals are paired entity bootstraps (2,000 resamples); effects are exploratory, without multiplicity correction.


## Llama minus Gemini: paired entity F1

| Comparison | Metric | n | Mean delta | 95% CI |
|---|---|---:|---:|---|
| Standard | f1 | 1000 | -0.0516 | [-0.0604, -0.0425] |
| Standard + taxonomy | f1 | 1000 | -0.0523 | [-0.0630, -0.0413] |
| CoT | f1 | 1000 | -0.0113 | [-0.0202, -0.0026] |
| CoT + taxonomy | f1 | 1000 | -0.0906 | [-0.0999, -0.0804] |
| Contrastive | f1 | 1000 | -0.0253 | [-0.0343, -0.0170] |
| Contrastive + taxonomy | f1 | 1000 | -0.1018 | [-0.1122, -0.0919] |
| Parent only | f1 | 1000 | -0.0318 | [-0.0419, -0.0215] |
| Sibling only | f1 | 1000 | -0.0501 | [-0.0603, -0.0397] |
| Children only | f1 | 1000 | -0.0739 | [-0.0842, -0.0643] |

## Difference in taxonomy uplift: Llama minus Gemini

| Comparison | Metric | n | Mean delta | 95% CI |
|---|---|---:|---:|---|
| Standard | f1_uplift | 1000 | -0.0007 | [-0.0126, +0.0114] |
| CoT | f1_uplift | 1000 | -0.0792 | [-0.0912, -0.0665] |
| Contrastive | f1_uplift | 1000 | -0.0765 | [-0.0895, -0.0643] |

## Children included minus children removed

| Comparison | Metric | n | Mean delta | 95% CI |
|---|---|---:|---:|---|
| Standard + taxonomy · F1 | f1 | 451 | +0.0123 | [-0.0029, +0.0278] |
| Standard + taxonomy · hierarchical F1 | hierarchical_f1 | 451 | -0.0306 | [-0.0467, -0.0152] |
| Standard + taxonomy · recall | gold_coverage | 451 | +0.0380 | [+0.0211, +0.0542] |
| Standard + taxonomy · most-specific hit | most_specific_hit | 451 | +0.0776 | [+0.0377, +0.1175] |
| CoT + taxonomy · F1 | f1 | 451 | +0.0086 | [-0.0069, +0.0233] |
| CoT + taxonomy · hierarchical F1 | hierarchical_f1 | 451 | +0.0050 | [-0.0085, +0.0186] |
| CoT + taxonomy · recall | gold_coverage | 451 | +0.0143 | [-0.0018, +0.0298] |
| CoT + taxonomy · most-specific hit | most_specific_hit | 451 | +0.0488 | [+0.0089, +0.0865] |
| Contrastive + taxonomy · F1 | f1 | 451 | -0.0170 | [-0.0304, -0.0035] |
| Contrastive + taxonomy · hierarchical F1 | hierarchical_f1 | 451 | -0.0003 | [-0.0133, +0.0127] |
| Contrastive + taxonomy · recall | gold_coverage | 451 | -0.0129 | [-0.0281, +0.0017] |
| Contrastive + taxonomy · most-specific hit | most_specific_hit | 451 | +0.0200 | [-0.0133, +0.0554] |

Child-removal comparisons use only the 451 child-present entities and reuse their full-context main-run responses.
Older incomplete Gemini outputs and the 20-entity Llama pilot are historical runs, not pooled into the final comparison.
Prediction-error review worksheets require real human annotation; outside-taxonomy labels are not automatically hallucinations.
