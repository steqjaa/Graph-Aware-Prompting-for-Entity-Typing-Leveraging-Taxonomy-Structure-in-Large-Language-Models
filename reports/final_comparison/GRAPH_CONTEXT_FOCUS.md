# Primary research comparison: graph-aware versus no graph

The primary contrast holds model, entities and prompting strategy fixed and adds oracle taxonomy context. Every comparison contains the same 1,000 entities. Positive deltas favor graph-aware prompting. Model comparisons are secondary; child removal tests a component of graph context.

Scores below are per-entity means. F1 is macro-F1. Intervals are 2,000-resample paired entity bootstrap intervals, exploratory without multiplicity correction. Printed values use four decimals; CSVs retain full precision.

## F1

| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |
|---|---|---:|---:|---:|---|
| Gemini | Standard | 0.3876 | 0.4582 | +0.0706 | [+0.0621, +0.0791] |
| Gemini | CoT | 0.3322 | 0.4735 | +0.1413 | [+0.1326, +0.1502] |
| Gemini | Contrastive | 0.3376 | 0.4739 | +0.1363 | [+0.1275, +0.1452] |
| Llama | Standard | 0.3360 | 0.4059 | +0.0699 | [+0.0599, +0.0801] |
| Llama | CoT | 0.3209 | 0.3829 | +0.0620 | [+0.0524, +0.0720] |
| Llama | Contrastive | 0.3123 | 0.3721 | +0.0598 | [+0.0502, +0.0698] |

## PRECISION

| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |
|---|---|---:|---:|---:|---|
| Gemini | Standard | 0.4683 | 0.5665 | +0.0982 | [+0.0880, +0.1081] |
| Gemini | CoT | 0.4659 | 0.6157 | +0.1497 | [+0.1383, +0.1615] |
| Gemini | Contrastive | 0.4741 | 0.6226 | +0.1485 | [+0.1372, +0.1604] |
| Llama | Standard | 0.3935 | 0.4801 | +0.0866 | [+0.0725, +0.1009] |
| Llama | CoT | 0.3709 | 0.4337 | +0.0628 | [+0.0501, +0.0759] |
| Llama | Contrastive | 0.3616 | 0.4375 | +0.0758 | [+0.0635, +0.0881] |

## RECALL

| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |
|---|---|---:|---:|---:|---|
| Gemini | Standard | 0.3500 | 0.4052 | +0.0552 | [+0.0466, +0.0635] |
| Gemini | CoT | 0.2721 | 0.4036 | +0.1314 | [+0.1229, +0.1402] |
| Gemini | Contrastive | 0.2751 | 0.4009 | +0.1258 | [+0.1174, +0.1340] |
| Llama | Standard | 0.3329 | 0.3873 | +0.0544 | [+0.0442, +0.0652] |
| Llama | CoT | 0.3140 | 0.3797 | +0.0656 | [+0.0550, +0.0766] |
| Llama | Contrastive | 0.3027 | 0.3568 | +0.0541 | [+0.0432, +0.0649] |

## MRR

| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |
|---|---|---:|---:|---:|---|
| Gemini | Standard | 0.3198 | 0.4674 | +0.1476 | [+0.1255, +0.1712] |
| Gemini | CoT | 0.2273 | 0.5302 | +0.3029 | [+0.2753, +0.3294] |
| Gemini | Contrastive | 0.2415 | 0.5324 | +0.2909 | [+0.2626, +0.3189] |
| Llama | Standard | 0.1565 | 0.1808 | +0.0243 | [+0.0049, +0.0429] |
| Llama | CoT | 0.1663 | 0.2017 | +0.0355 | [+0.0163, +0.0562] |
| Llama | Contrastive | 0.1810 | 0.2088 | +0.0278 | [+0.0063, +0.0485] |

## NDCG

| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |
|---|---|---:|---:|---:|---|
| Gemini | Standard | 0.3250 | 0.4731 | +0.1480 | [+0.1363, +0.1600] |
| Gemini | CoT | 0.2645 | 0.5096 | +0.2451 | [+0.2318, +0.2586] |
| Gemini | Contrastive | 0.2741 | 0.5073 | +0.2332 | [+0.2190, +0.2469] |
| Llama | Standard | 0.2348 | 0.3727 | +0.1379 | [+0.1264, +0.1487] |
| Llama | CoT | 0.2249 | 0.3586 | +0.1337 | [+0.1221, +0.1450] |
| Llama | Contrastive | 0.2295 | 0.3469 | +0.1175 | [+0.1058, +0.1287] |

See figures 01 (F1), 16 (precision), 03 (recall), 17 (MRR), 04 (nDCG), and 08 (paired gains) in both model figure directories. Taxonomy context is oracle context.
