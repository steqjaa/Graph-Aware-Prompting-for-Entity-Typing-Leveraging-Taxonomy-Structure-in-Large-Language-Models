# Historical literature notes

Sources checked 23 September 2026. This is a targeted literature review. Searches combined entity typing, taxonomy, parent/sibling/child context, ablation, and the titles in the exposé. Primary sources included ACL Anthology, PMLR, arXiv author manuscripts, and institutional publication records. Full citations: [references.md](references.md).

| Paper | Relevant evidence | Role in the study |
|---|---|---|
| Peng et al. (2024), WiKC [1] | Automatically refined taxonomy; includes downstream entity-typing evaluation. | Establish the dataset's origin, not that this prompting method will work. |
| Shen et al. (2025), TaxoInstruct [2] | NoParentPretrain and NoSiblingPretrain ablations examine transfer between sibling and parent skills; sections 4.1-4.3. | Closest evidence against an unqualified “no previous work” claim. Training-skill ablation differs from prompt-context ablation. |
| Xia et al. (2024), ATTEMPT [3] | Section 4.4 and Tables 4-5 remove path nodes in parent finding and grandchild information in child finding. | Supports studying relation utility by task stage; not a universal ranking of relation types. |
| Ma et al. (2026), DR-FET [4] | Retrieves candidate types using entity/type descriptors for fine-grained entity typing. | Current comparator for realistic candidate retrieval; not proof that the existing MiniLM retrieval is equivalent. |
| Wei et al. (2022) [5]; Chia et al. (2023) [6] | Reasoned demonstrations and positive/negative demonstrations motivate prompt families. | Cite the design inspiration. The thesis variants are adaptations, not exact reproductions. |
| Järvelin and Kekäläinen (2002) [7] | Graded relevance and cumulative-gain evaluation. | Justifies the ranking metric family. It does not justify depth/max-depth gains automatically. |
| Shi et al. (2023) [8]; Liu et al. (2024) [9] | Irrelevant context and evidence position can affect model use of context. | Motivates controls and compact wording; behavioural equivalence requires experiments. |
| Jiang et al. (2023), StructGPT [10] | Separates reading structured evidence from reasoning in question answering. | Background for evidence acquisition versus use, not direct entity-typing evidence. |
| Ghamlouch and Alam (2025), Taxoria [11] | Enriches an existing taxonomy with generated and validated candidates. | Related taxonomy-maintenance work; use a narrow task distinction. |
| Pan et al. (2024) [12]; Sun et al. (2024) [13] | Survey of LLM/KG integration and graph-guided reasoning, respectively. | Broad framing; prioritize the more directly relevant works above for the contribution. |

## Relation to prior work

Related evidence exists, but the experimental interventions differ. TaxoInstruct concerns jointly learned parent/sibling generation skills. ATTEMPT studies local information in taxonomy completion stages. Neither demonstrates that one relation should always be most useful for ranked entity typing. A defensible claim is: “This work examines parent, sibling, and child context under a shared prompting and evaluation protocol for entity typing, and tests whether their utility depends on class structure and context availability.”

Do not infer causal task dependence merely because different papers use different ablations. To make that claim within this project, compare tasks under matched conditions. With one task, conclude only about entity typing and distinguish differences by depth/availability from differences across tasks.

## Bibliographic notes

| Existing entry | Corrected entry or treatment |
|---|---|
| “Kim ... Contrastive Chain-of-Thought Prompting” | Chia, Chen, Luu, Poria, and Bing (2023), arXiv:2311.09277 [6]. |
| “Hao ... StructGPT” | Jiang, Zhou, Dong, Ye, Zhao, and Wen (2023), EMNLP [10]. |
| “Peng [initials not provided]” | Yiwen Peng, Thomas Bonald, and Mehwish Alam (2024) [1]. |
| “Ghamlouch, M.” | Zeinab Ghamlouch and Mehwish Alam [11]. Use ECAI 2025 demo-track provenance; web mirrors show inconsistent later dates. |
| “Liu et al. (2023), TACL” | Cite the 2024 TACL publication [9]; 2023 refers to the preprint chronology. |
| “Sun et al. (2023), ICLR 2024” | Use 2024 for the conference paper [13]; distinguish its 2023 preprint. |

No unsupported full author placeholders remain in the new bibliography. Sources were checked at the abstract/metadata level except ATTEMPT and TaxoInstruct, whose relevant methods/ablation passages were inspected. Do not describe this as a full reproduction of the cited studies.
