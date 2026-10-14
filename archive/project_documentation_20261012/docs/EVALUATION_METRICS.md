# Evaluation definitions (schema version 2)

These definitions take precedence over older metric descriptions for the final experiment. Every condition uses the same full annotated gold set. Label normalization remains the historical lowercasing, whitespace, underscore and surrounding-punctuation normalization, without semantic aliases.

## Primary performance

For each entity, P and G are sets of unique normalized predicted and gold labels. TP is their intersection size. Precision = TP/|P|; recall = TP/|G|; F1 = 2TP/(|P|+|G|). Empty predictions receive zero precision, recall and F1. The manifest has nonempty gold sets.

- **Macro-F1:** arithmetic mean of entity F1. This is explicitly the per-entity definition used in this evaluation.
- **Loose Macro-F1:** harmonic mean of mean entity precision and mean entity recall, reported separately as `loose_macro_f1` to distinguish the conventional alternative aggregation.
- **Micro-F1:** 2ΣTP/(Σ|P|+Σ|G|). Micro precision and recall use the corresponding pooled denominators.
- **Most-specific hit:** any prediction matches any maximum-depth annotated gold label, using the historical implementation. This depth-based definition is not a claim about a unique graph-minimal target in a DAG.
- **Strict accuracy:** P equals G exactly.

Gold coverage is entity recall. Extra labels reduce precision/F1. Duplicate labels do not count more than once in set metrics, but output-length and historical rank metrics retain the original prediction list.

## Hierarchical F1, not LCA-F1

The implementation uses **ancestor-augmented hierarchical F1**. It does not implement or approximate the separate Kosmopoulos LCA-F1 algorithm. The requested text offered “LCA / hierarchical F1”; this release chooses the directly auditable ancestor-closure version. Do not label the reported values LCA-F1 in the thesis.

WiKC edges point from child to parent. Augment each resolved predicted and gold class with all reachable ancestors and itself; include roots; calculate precision/recall/F1 over these augmented sets. Average the resulting entity metrics. All DAG parents are considered, not just a chosen path. The scorer rejects cyclic taxonomies.

Class mapping is exact normalized-label matching against `wiki_taxonomy_labels.tsv`. A label matching annotated gold resolves to the annotated class IDs. Otherwise only a unique class-label mapping resolves. Ambiguous and unmapped labels are retained as opaque false-positive items in the augmented predicted set; they do not disappear from the denominator. Mapping ambiguity, candidate IDs and resolved IDs are recorded for audit. Exact-label disambiguation is an evaluation convention and should be disclosed.

Common ancestors can inflate hierarchical scores; interpret them alongside flat F1 and error distance. Hierarchical values cannot be numerically compared with implementations that exclude roots, choose one parent, or use a different ambiguity policy.

## Error distance and direction

For an incorrect, resolved prediction p, distance is the minimum unweighted shortest-path length to any annotated gold class in the **undirected** WiKC graph. Means and medians in `taxonomy_error_distance.csv` pool measurable incorrect predictions within each condition. `taxonomy_error_mean` in entity scores is the within-entity mean, and averaging it gives entities with measurable errors equal weight. These are different estimands.

Unknown labels, ambiguous labels, labeled IDs absent from the graph, and disconnected components have **no finite distance**. They are reported separately and excluded from numeric distance summaries; the count of unmeasurable errors is explicit. Never silently substitute zero or a fixed large distance. The nearest gold set includes broad annotated gold labels, which can shorten distances; this is the requested nearest-G measure, not distance to only the deepest target.

Exclusive categories use this precedence: exact normalized gold > broader > narrower > sibling > other taxonomy. Broader means reachable upward from a gold node; narrower means a gold node is reachable upward from the prediction. Distances in those buckets use directed paths, with broader 1/2/3+ and narrower 1/2+. Siblings share an immediate parent and are neither broader nor narrower under this precedence. A DAG prediction can relate to multiple gold branches; the precedence resolves overlaps. Unresolved/disconnected cases remain separate. Distributions provide count and share denominators.

## Hierarchy consistency

For each resolved predicted graph node, all reachable ancestors, including roots, must appear in the **unaugmented** mapped prediction set. Node consistency = number satisfying this condition / number of resolved predicted graph nodes. Set consistency = one if every resolved predicted graph node satisfies it, zero otherwise. Entities with no resolved graph nodes are undefined, not perfectly consistent. Summary files include the number of eligible entities. Unknown predictions are reported separately; consistency is not correctness and does not certify their validity.

## Paired inference

All comparisons align by entity ID and require identical gold annotations. Model-system comparisons also require identical input fingerprints. Child-removal comparisons require exactly the main run's child-present cohort and matching source prompt hashes. No silent intersection or dropped unmatched entities is allowed.

Graph versus no graph, relation-only pairs, model pairs, and children-included versus removed use the same entity's two scores. Percentile bootstrap confidence intervals resample these paired entity differences 2,000 times using seed 230917. Report mean delta and 95% CI with helped/tied/harmed counts (absolute delta ≤ 1e-12 is a tie). Intervals are exploratory, not corrected for multiple comparisons. The model-uplift comparison bootstraps each entity's difference of graph deltas.

Depth uplift uses one paired entity observation, grouped by maximum annotated gold depth 4–5, 6–7, or 8+. Lower-depth exclusions and per-bucket support are reported. This is **target-depth stratification**, not inferred ancestor distance. Child-availability strata alone are not a matched causal child-removal experiment; only the generated 451-entity intervention is used for that comparison.

## Supporting and qualitative outputs

MRR and depth-weighted nDCG retain historical behavior for comparability and are supporting metrics. The rank plot bins first deepest-gold recovery into Rank 1, Rank 2, Rank 3+, and Miss. The manual worksheet and representative helped/harmed cases support qualitative review; annotation completeness is not assumed.

## References and implementation sources

- Choi et al. (2018), *Ultra-Fine Entity Typing*: https://aclanthology.org/P18-1009/
- Kosmopoulos et al., *Evaluation Measures for Hierarchical Classification: a unified view and novel approaches*: https://arxiv.org/abs/1306.6802 (hierarchical metric distinctions; LCA-F1 is a separate algorithm).
- Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*: https://aclanthology.org/P18-1128/
- Plaud et al. (2024), *Revisiting Hierarchical Text Classification: Inference and Metrics*: https://aclanthology.org/2024.conll-1.18/
- Hugging Face, chat templates: https://huggingface.co/docs/transformers/v4.43.4/chat_templating
- Hugging Face, generation guidance: https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_tutorial.md
