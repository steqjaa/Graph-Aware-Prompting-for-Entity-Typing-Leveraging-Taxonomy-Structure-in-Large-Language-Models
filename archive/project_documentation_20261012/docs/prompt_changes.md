# Prompt cleanup: what changed and what did not

The new v4 wording is a proposed intervention, not a retroactive change to old experiments. Its design retains reasoned and contrastive demonstrations [5,6] and warnings against unsupported context copying [8]. Citations refer to [references.md](references.md). References belong in the thesis documentation rather than the model input, where they would themselves change the experimental condition.

## Shared task block

> List valid entity-type labels, most specific first, then broader valid types. Use concise labels, not entity names or descriptions. Include only evidence-supported types; do not repeat labels or pad the list.

## Graph block

> The context surrounds an omitted target type; it is evidence, not answer options. Parents are broader, children narrower, and siblings share a parent. Infer the type using the description and these relations. Copy a neighbour only if valid for this entity. Siblings need not be disjoint; predict a child only when the description independently supports it.

The remaining concise instruction blocks are in `prompts_v4/instruction_blocks.json`. Demonstration text is intentionally retained because changing or removing it would confound wording compression with demonstration changes.

## Review-set invariants

For all 260 review prompts, an assertion checks that everything from `Example:` onward is byte-identical after replacement. This preserves demonstration entities, positive and negative example outputs, target entity descriptions, all supplied context, and the final answer cue. Original and new SHA-256 hashes, word counts, and character counts are recorded in `analysis/prompt_length_audit.csv`. JSON output-schema instructions are outside the replaced blocks and remain unchanged.

| Primary condition | Mean original words | Mean v4 words | Reduction |
|---|---:|---:|---:|
| Direct | 131.0 | 83.0 | 36.6% |
| Direct + oracle | 293.6 | 192.6 | 34.4% |
| CoT | 260.0 | 165.0 | 36.5% |
| CoT + oracle | 516.6 | 311.6 | 39.7% |
| Contrastive | 362.0 | 233.0 | 35.6% |
| Contrastive + oracle | 628.6 | 382.6 | 39.1% |

Word counts are whitespace-delimited and are not tokenizer counts. Shortening removes repeated instructions but can still alter behaviour. Compare old versus v4 on development entities using identical decoding and document changes in parse success, specificity, coverage, and token use before finalizing v4.

## Generated fresh cohort

The 100-entity cohort is generated using the new instruction blocks with freshly selected capped context. It is not a byte-preserving review copy: it intentionally changes entities and neighbour selection. The preserved demonstration is rendered with the same neighbour cap. All 900 prompts are NOT_RUN. No old result should be copied into these folders.

The review manifest combines two source sets and can repeat an entity/condition key across future extensions. It is intended for inspection. Use the fresh-cohort manifest for a new nine-condition experiment. Run manifest-relative paths from the cleaned project root.
