# Final experiment evaluation

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

Hierarchical F1 uses ancestor closure, including self and roots; it is not LCA-F1.
Gemini and Llama use different inference systems and decoding settings. The comparison does not isolate architecture.
Confidence intervals are paired entity bootstraps (2,000 resamples); effects are exploratory, without multiplicity correction.

Llama main and child-removal inference have not been run here. No Llama comparison is reported.

Child-removal comparisons use only the 451 child-present entities and reuse their full-context main-run responses.
Older incomplete Gemini outputs and the 20-entity Llama pilot are historical runs, not pooled into the final comparison.
Prediction-error review worksheets require real human annotation; outside-taxonomy labels are not automatically hallucinations.
