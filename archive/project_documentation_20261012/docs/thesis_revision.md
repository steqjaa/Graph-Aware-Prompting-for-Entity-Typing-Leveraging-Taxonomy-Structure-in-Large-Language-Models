# Graph-aware prompting for entity typing
## Historical pilot analysis

Historical pilot analysis, 23 September 2026. Citations refer to references.md and references.bib.

## 1. Introduction: a study of structural evidence

Entity typing assigns semantic categories to entities. When those categories belong to a taxonomy, the task also involves choosing an appropriate level of specificity and distinguishing nearby concepts. This thesis investigates how supplying taxonomy structure changes the predictions of a large language model. Its purpose is to explain which structural evidence is useful, under what conditions, and with what failure modes.

WiKC provides a refined Wikidata taxonomy and a relevant entity-typing setting [1]. The broader integration of language models and knowledge graphs motivates studying how structured evidence is presented at inference time [12]. However, the availability of a graph does not by itself establish that every neighbourhood is helpful. Parent, sibling, and child relations convey different constraints, and context selected using the correct answer represents a different problem from context retrieved without it.

The empirical study begins with exploratory prompting approaches. Direct prompting provides a reference for predictions from the entity description. Reasoned prompting requests an explanation before the ranked output, while a contrastive variant includes a negative example and asks the model to reject unsupported alternatives. These choices are motivated by chain-of-thought and contrastive prompting research [5,6]; their usefulness for entity typing is tested rather than assumed.

The resulting main comparison crosses the three prompting styles with the absence or presence of oracle taxonomy context. Further analyses separate early recovery of the most specific annotated type, recovery of broader gold types, and output length. Relation-specific ablations and retrieval-based context then address why observed changes occur. This sequence connects method selection to the questions raised by the discovery phase.

## 2. Research questions

1. How does taxonomy context change specific-type recovery, broader-type coverage, and prediction length?
2. Which parent, sibling, or child evidence is useful when its availability and quantity are controlled?
3. How do the effects vary between leaf and non-leaf classes and across taxonomy depth?
4. How do oracle and retrieved context differ, and what errors originate in retrieval?
5. How sensitive are predictions to incorrect structural context?

A comparison between WiKC and raw Wikidata remains an additional planned question from the exposé. It is not presented as completed in this revision.

## 3. From discovery to the six-condition comparison

The initial implementation explored several prompt styles and versions. The first lesson is that the term “graph-aware” must identify how the context was obtained. In the direct graph condition, neighbours are selected around the gold most-specific class. This is an oracle-assisted diagnostic even though the class name is omitted. Retrieved variants instead use the entity and description to select candidate neighbourhoods, but also differ in centre visibility and the amount of supplied context.

The second lesson is that a single aggregate score obscures distinct behaviours. A model may recover more ancestors without finding a more specific label earlier. It may also return more labels, increasing opportunities for a gold match. For this reason, the main six conditions are evaluated with separate views of MRR, nDCG, most-specific recovery, and gold coverage, alongside output-length diagnostics.

The third lesson concerns relation availability. Eight of the twenty pilot entities have no child context. An all-entity average therefore mixes exposure to child evidence with an empty-context case. The subsequent protocol reports availability explicitly and prepares a child-present cohort for controlled analysis.

The six retained conditions form an interpretable 3-by-2 comparison: direct, CoT, and contrastive prompts, each without context and with oracle context. This design allows within-style comparisons. It should not be described as the six incremental graph levels proposed in the exposé; relation and ancestor-path interventions are a separate axis of experimentation.

## 4. Evaluation rationale

MRR measures how early the first deepest annotated gold type appears. It is useful when early specific-type identification is the primary objective, but it does not reward subsequent recovery of other valid types. nDCG measures graded ranking quality across the output [7]. The current implementation assigns relevance proportional to annotation depth, so its result depends on the assumption that deeper gold types deserve more credit. Gold coverage measures the fraction of the annotated set returned, and most-specific hit rate reports whether a deepest gold type appears at a specified cutoff.

These measures should be interpreted jointly. For example, higher coverage with unchanged MRR can indicate additional broad gold types rather than improved early specificity. Common-cutoff analyses help identify how much the conclusions depend on list length. Exact matching against the annotations also limits interpretation: an unmatched prediction is not necessarily semantically false.

## 5. Observed pilot findings

The principal saved run uses meta-llama/Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`, local Transformers, bfloat16, and an NVIDIA H100 NVL. It contains twenty entities and ten conditions. All 200 completed raw results were checked against the saved MRR, nDCG, and gold-coverage values before producing the new figures. The six-condition subset is reported below; retrieved and shuffled variants remain separate controls.

| Prompt | MRR | nDCG, full output | Deepest-gold hit, anywhere | Gold coverage | Mean types/entity |
|---|---:|---:|---:|---:|---:|
| Direct | 0.250 | 0.315 | 25% | 40.75% | 7.60 |
| Direct + oracle | 0.300 | 0.471 | 35% | 45.23% | 8.15 |
| CoT | 0.225 | 0.283 | 25% | 37.03% | 7.75 |
| CoT + oracle | 0.239 | 0.454 | 35% | 48.48% | 9.05 |
| Contrastive | 0.225 | 0.215 | 25% | 23.58% | 6.20 |
| Contrastive + oracle | 0.283 | 0.439 | 35% | 44.00% | 8.10 |

Source: supplied raw results and recomputation in `analysis/pilot_condition_summary.csv`. These are historical-prompt results, not results for the shortened prompts.

Oracle context raises mean gold coverage by 4.48, 11.45, and 20.42 percentage points for direct, CoT, and contrastive prompting, respectively. The corresponding MRR increases are 0.050, 0.014, and 0.058. Every paired MRR bootstrap interval includes zero. The pilot therefore supports a descriptive pattern of greater gold recovery, while uncertainty remains substantial for early specific-type recovery.

The large contrastive coverage change begins from a relatively weak no-context baseline; it does not make the contrastive oracle condition the strongest absolute performer. CoT with oracle context has the highest full-output coverage among the six, whereas direct prompting with oracle context has the highest MRR and nDCG. This is a useful disagreement between evaluation views rather than a reason to select whichever metric favours one method.

Oracle conditions also return more types on average. At cutoff five, gold coverage is 19.57% versus 24.62% for direct prompts, 17.43% versus 22.03% for CoT, and 14.62% versus 22.15% for contrastive prompts. Thus the descriptive coverage pattern remains at a common cutoff, although this alone does not identify its causal mechanism.

Retrieved context does not reproduce the oracle pattern consistently. Direct gold coverage changes from 40.75% to 35.85%, CoT from 37.03% to 35.43%, and contrastive from 23.58% to 28.86%. The interpretation must consider candidate quality and rendering differences, not simply conclude that graph structure succeeds or fails.

The child-present subgroup contains twelve entities. Within this subgroup, full oracle context changes gold coverage by 4.13, 14.52, and 18.83 percentage points for the three styles. These are full-context comparisons conditional on child availability; they do not isolate a causal child-only effect. The separate archived relation-only run is retained descriptively because its model/backend metadata differ from the principal run.

## 6. Related work and the contribution boundary

Prior research already offers evidence that structural information matters differently across procedures. ATTEMPT separates parent finding from child labelling and ablates path and grandchild information within those stages [3]. TaxoInstruct evaluates parent-finding and sibling-finding pretraining across expansion and construction tasks and finds benefits from their joint use [2]. Neither study is a direct parent-only versus sibling-only versus child-only prompt ablation for this entity-typing setup.

The defensible contribution is therefore a controlled account of relation utility for ranked entity typing under explicit context regimes and budgets. It is not a claim that task-dependent structural information has never been studied. A further task would be needed to establish cross-task variation experimentally within this thesis.

Recent descriptor-based retrieval for fine-grained entity typing provides a particularly relevant comparison for realistic candidate selection [4]. Taxoria concerns taxonomy enrichment rather than the same downstream typing task [11]. Work on irrelevant context and long-context position motivates careful context controls [8,9], but does not establish that shorter prompts will preserve this model's behaviour.

## 7. Next experiments and limitations

The revision prepares 100 previously unexposed, child-present entities and nine conditions per entity: the six main conditions plus direct parent-only, sibling-only, and child-only prompts. All are marked NOT_RUN. Neighbour counts are capped at three per relation, and actual availability and selected class IDs are recorded. Because full context may contain more nodes than a relation-only arm, a matched-total-budget control remains necessary for claims about complementary information.

The shorter prompts preserve the original demonstration blocks, contextual evidence, negative examples, and response schemas in the review set. The six primary prompt families are approximately 34-40% shorter by whitespace-delimited word count. Behavioural equivalence is not established; these changes define a new prompt version requiring evaluation on development data before a frozen final test.

Further limitations include the twenty-entity pilot size, a single principal model, label-normalization sensitivity, depth-based relevance assumptions, and possible annotation incompleteness. Taxonomy-quality comparisons, ancestor-path-only interventions, and broad cross-task claims remain future work. These questions were not tested in the historical pilot.

## References

Bibliographic records: [references.md](references.md) and [references.bib](references.bib). Pilot values are derived from the archived experiment outputs.
