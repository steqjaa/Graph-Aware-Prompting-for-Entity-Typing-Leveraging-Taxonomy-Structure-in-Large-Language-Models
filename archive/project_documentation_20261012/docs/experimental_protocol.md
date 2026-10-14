# Historical experimental design notes

Design notes, 23 September 2026. These notes document an earlier proposed design; executed configurations are recorded with each result set. Numbered citations resolve to [references.md](references.md) and [references.bib](references.bib).

## Research objective

Explain when taxonomy context changes entity typing, which errors it resolves or creates, and how that depends on context source, relation availability, and prediction specificity. Metric improvements are evidence for these questions, not the objective themselves. WiKC supplies the taxonomy and motivates downstream entity-typing evaluation [1].

## Separate three stages

1. **Discovery:** describe the early standard, graph-aware, CoT, and contrastive variants; explain what their failures taught us. These are development observations, not independent confirmation.
2. **Six-condition comparison:** cross direct / CoT / contrastive prompting with no context / oracle context. This is the six-condition design present in the saved supervisor-v3 run. The exposé also proposes six *context levels* (baseline, parent, path, sibling, child, combined). These are different designs; do not silently equate them.
3. **Structural explanation:** hold direct prompting fixed and compare none, parent-only, sibling-only, child-only, and full context on the same fresh entities. Ancestor-path-only is a further planned condition, not implemented in the delivered nine-condition cohort. Add retrieval and noise controls in separate, clearly labelled experiments.

The six-condition set is retained because it separates reasoning style from supplied structure, not because it maximized pilot scores. This rationale is proposed for the thesis; the attachments do not document every historical decision.

## Cohorts and estimands

The supplied file contains 1,000 entities. Excluding all 61 entity IDs exposed in attached prompts/results or used as a demonstration leaves 873 usable entities under the existing single-most-specific-target and graph-context criteria. Of these, 401 have direct children; 388 have all three relations. The new sample takes 100 from the 401 using seed 20260923. It is a **child-present oracle diagnostic cohort**, not a representative estimate for all entities.

Report a full eligible sample and a child-present sensitivity analysis separately. Filtering out leaves answers how child evidence behaves *when it exists*; it cannot establish the average benefit over the full taxonomy. Do not infer a relation's causal contribution merely from the difference between leaf and non-leaf groups, which also differ in class and depth.

The generator excludes all entity IDs visible in the attachments. If additional experiments were run elsewhere, add their IDs before treating this pool as untouched. No claim of globally unseen data is made.

## Control counts explicitly

Three quantities need separate controls:

- **Entity count:** `--count`, fixed across conditions, with a saved ID list and seed.
- **Neighbour count:** `--neighbors`, a maximum per relation, including parents. The default is 3. `--require all --exact` requires at least that many neighbours in each relation and then selects exactly that many. This can sharply restrict the population; inspect the audit before choosing it.
- **Prediction count:** keep evidence-supported variable output length, and evaluate common cutoffs k = 1, 3, 5, 10. Do not force unsupported extra predictions just to obtain equal list lengths.

A cap of 3 per relation means up to 9 nodes in full context versus up to 3 in a single-relation condition. It does **not** control the total information or token budget. Use the single-relation arms for equal maximum-node comparisons; before attributing a full-context gain to complementarity, add a matched-total-node or matched-token control. Record actual counts and tokenizer-measured input/output tokens. Current prompt length audit uses words and characters, not tokens.

Selection within each neighbourhood is deterministically shuffled using a seed derived from the entity's target class and relation. Budgets are nested prefixes of that ordering. Repeat neighbour seeds in a planned robustness analysis; a single seed cannot estimate neighbour-selection variance.

## Oracle, retrieval, and noise

Oracle prompts use the hidden gold class to find its neighbours. Omitting the class name does not remove this privileged information. These prompts diagnose available structural evidence; they do not model inference when the target is unknown. Use entity and description only to retrieve candidates in a deployment-style experiment, and report candidate recall before typing. DR-FET provides recent, directly relevant retrieval-based entity-typing work [4].

The archived retrieved variants also expose candidate centres and multiple neighbourhoods, whereas the oracle variants omit the target. Therefore their comparison changes more than retrieval correctness. To isolate retrieval error, use an identical rendering format, centre visibility, candidate count, and context budget.

Incorrect-context experiments need the same budget and formatting as the corresponding correct-context arm, with accidental correct overlaps measured. Evidence on irrelevant context motivates the control, but does not prove the thesis's noise hypothesis [8]. The existing shuffled control leaves direct-prompt MRR unchanged in this pilot.

## Metrics: which question each answers

| Measure | Question | Limitation |
|---|---|---|
| MRR (deepest annotated gold) | How early is the first deepest-gold match? | Ignores all later valid types; matching any one of several deepest gold types suffices. |
| Depth-weighted nDCG@k | How well are graded gold types ranked in the top k? | Gain = depth/max gold depth is this project's assumption, not a standard entity-typing relevance judgment. |
| Most-specific Hit@1 / Hit@k / Hit-anywhere | Is a deepest gold type recovered at the stated cutoff? | Anywhere depends on output length; it does not measure full coverage. |
| Gold recall / coverage | How much of the annotated type set is recovered? | Can rise with longer lists or generic ancestors; gold may omit valid types. |
| Precision@k and output length | How many early predictions are gold matches, and how much does the model output? | Exact-label precision can penalize valid paraphrases. |

MRR is preferable **if the primary question is early identification of a specific type**. nDCG is preferable for graded multi-type ranking. Neither is universally better. Järvelin and Kekäläinen [7] establish the graded cumulative-gain approach; the depth gain here requires its own rationale and sensitivity checks. In the pilot, nDCG and coverage expose changes hidden by nearly unchanged MRR.

Historical scoring defines “most specific” by maximum supplied depth. Cohort selection uses graph minimality among the entity's annotated types. These can disagree in a DAG. The supplied figures retain historical scoring for comparability; explicitly evaluate the graph-minimal gold set in the confirmatory study rather than relabelling old scores.

Depth-zero gold types receive zero relevance when deeper types exist, although they still count in gold coverage. Exact matching uses the historical case/whitespace/punctuation normalizer, with no added aliases. Any future synonym/QID mapping must be frozen independently of the held-out outputs and reported as a separate sensitivity analysis. An unmapped type is not automatically a hallucination.

## Estimation and inference

Pair conditions by entity. Report means, paired differences, improved/worsened/tied counts, and intervals. The delivered pilot intervals use 5,000 entity bootstrap resamples (seed 33). They describe this small sample and do not account for model, prompt, or taxonomy variation. They are exploratory, not multiplicity-adjusted significance tests. For a new study, preregister primary comparisons and cutoffs; do not choose a metric after inspecting whichever gains most.

Use one model revision, tokenizer, backend, precision, decoding configuration, and output budget across compared conditions. Record raw responses, parse status, stop reason, truncation, and failed calls. Do not silently exclude failures. The archived relation-only run has different backend/model metadata from the H100 bf16 run; it must not be pooled with that run's baseline to claim causal parent/sibling/child effects.

## Claims that remain untested

The supplied experiments do not establish WiKC superiority over raw Wikidata, a general depth interaction, a universal best relation, or cross-task generality. WiKC-versus-Wikidata requires aligned entities and labels, comparable retrieval and budgets, and a frozen evaluation target. “Information gain” should mean empirical predictive utility unless an explicit entropy or mutual-information estimator is implemented. One entity-typing task can yield task-specific findings, but demonstrating that relations differ *across tasks* requires another task or a carefully qualified comparison with prior studies [2,3].
