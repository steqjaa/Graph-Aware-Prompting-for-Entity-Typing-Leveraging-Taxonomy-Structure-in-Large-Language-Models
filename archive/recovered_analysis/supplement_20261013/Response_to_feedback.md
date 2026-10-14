# Response to thesis feedback

All 45 written annotations were reviewed. The original PDF also contains highlights and line annotations; these were used to locate the associated passages. IDs below follow annotation order in the original PDF, so they are not consecutive.

The revision uses only the supplied external-demonstration experiment outputs. It contains 36,000 main tasks, 1,353 ablation tasks, fixed-denominator and accepted-output analyses, and a focused source-supported semantic review.

| Comment | Original page | Revised location | Action and limits |
|---|---:|---|---|
| C2 | 1 | Title and Contents | Main title/subtitle separated; hierarchical chapter/subsection navigation provided. |
| C4 | 1 | Title | Graph-Aware spelling applied. |
| C6 | 2 | Abstract; 1.1; 5.2; 7.4 | Treatment explicitly includes structural evidence, instructions, and demonstration presentation. |
| C8 | 3 | 5.3; References | Stable model ID, Gemini API setting, run dates, official model page and model card supplied. No undocumented architecture inferred. |
| C10 | 4 | 1.1–1.2 | Introduction numbered and thesis outline added. |
| C12 | 4 | 1.1; 4.3; 7.5 | Sony Alpha 230 replaces the occupation-copying example; centre label absent from its description. Recomputed 265/735 lexical split and all model/style gains. |
| C14 | 4 | 1.2; 5.4 | Contributions labelled C1–C3. Development history condensed after final method. |
| C16 | 5 | 1.2; 7; 9; 10 | RQ1–RQ4 explicitly labelled and answered. |
| C18 | 5 | 1.2; 7.4–7.5 | H1 and H2 stated in advance and tested by name; neither supported as a general claim. |
| C20 | 6 | 2; 4–6 | Preliminaries rewritten with numbered definitions, no empirical results. Centre, graph-minimal set, and maximum-depth target distinguished. |
| C22 | 6 | 2.3–2.5 | Shortest- and longest-path depth may differ; centre and ranking target related formally. |
| C23 | 6 | Definitions 2.1–2.11 | All listed notions formalized, including orientation, relations, closure, entity/gold, minimal and deepest classes, tie-break, capped context, output task, normalization/resolution, and condition. |
| C25 | 7 | 2.4; 4.2; 6.2 | Gold is the listed set, not closure by definition; 16 non-closed sets recomputed. Flat and closure scoring implications explained. |
| C27 | 7 | 2; 5–8 | Prompt and decoding details moved to methods, metric implementation to evaluation, numerical outcomes to results. |
| C28 | 8 | 2.3; 3.2 | In-context terminology corrected. |
| C32 | 8 | 2.9; 5.1 | All conditions described as one-shot with one demonstrated entity. |
| C34 | 8 | 5.1; validation materials | Verified external demonstration Rod Fergusson is outside the evaluation cohort and present in all main prompts. |
| C36 | 9 | 2.9; 3.2 | Chain-of-thought terminology standardized. |
| C39 | 9 | 3.2; References | Turpin et al. (2023) cited; visible explanations not equated with faithful internal reasoning. |
| C42 | 11 | 3.1–3.2; Table 3.1 | Added Xu and Barbosa, Chen et al., Onoe et al., ConnectE, TET, LLM-based typing, Wikidata citation, and positioning table. |
| C45 | 12 | 3.2 | Positive one-sentence research gap supplied. |
| C47 | 13 | 4.2; Tables 4.1–4.2 | Recomputed class/edge/entity counts, mean gold size, empty and truncated lists for every relation. |
| C51 | 14 | 2.5; 4.2 | Lexicographic string tie-break verified for every centre. 136 maximum-depth ties and 89 centre-outside-target cases reported. |
| C53 | 14 | 2.5; throughout | Centre label defined and used consistently; undefined reference-type terminology removed from prose. |
| C54 | 16 | 4.1–4.3 | Consecutive subsection numbering restored. |
| C56 | 16 | 4.3 | Unique mode identified as depth 7; depths 6 and 7 jointly 46.5%. |
| C58 | 17 | 5.1–5.4; Figure 5.1; Appendix C | Final pipeline and extraction algorithm precede templates, models, validation and condensed history. Nine complete prompts printed. |
| C60 | 19 | 5.2; 7.4; prompt diffs | Exact string differences supplied. All 549 empty-child entities verified; instruction-and-format effects reported for Gemini and every Llama seed. |
| C62 | 20 | 5.3; Appendix B | Provider-default thinking setting, active internal thinking, 65,536 output limit and per-condition token means reported. Style/model interpretation qualified. |
| C63 | 20 | 5.3 | Full checkpoint revision included; archived vLLM version checked against official 5 October 2026 release. Hardware updated to H100 from new run metadata. |
| C65 | 20 | 5.3; 7.3 | T=1.0 justified as protocol choice, not optimum. Seeds 0/42/123 and ablation seed 0 explicit; mean and sample SD reported. |
| C67 | 21 | 5.4; Appendix B; retry exports | Recorded policy, first-call validity, retry counts, failure counts and conditional acceptance explained. Gemini transport cap cannot be recovered; observed maximum five calls disclosed instead of inventing a configured maximum. |
| C68 | 21 | 5.4; Appendix B | Gemini timeouts and HTTP 429, recovery sessions, first-valid retention and no score-based selection described using new archive. |
| C69 | 22 | 6.2; Table 6.1 | Worked flat, ranking and hierarchical metric example added. |
| C70 | 26 | 5; 7–8 | Figures explicitly referenced and interpreted in surrounding text. |
| C71 | 27 | 7.1–7.2 | Graph/no-graph results consolidated into joint comparison; paired-effect interpretation kept adjacent for readability. |
| C72 | 27 | 7.7; Figure 7.4 | Exact labelled mean type counts provided for all primary conditions and runs; all nine exported. |
| C74 | 29 | 7.1–7.5 | Mechanistic interpretations added, distinguished from measured internal causes; instruction control incorporated. |
| C75 | 30 | 7.6 | Ablation motivation explained. Additional audit found child-related wording also changes; both evidence and instruction removal are disclosed. |
| C76 | 30 | Figure captions | Unnecessary source wording removed. |
| C77 | 30 | Throughout; 5.1; Appendix C | Direct is the consistent prose term; original standard file names retained only for exact provenance. |
| C81 | 31 | 7.6–7.7; 8 | Ablation, length, precision/recall and semantic patterns interpreted with limits. |
| C82 | 32 | 8.3; Table 8.2 | Actual output examples presented in an evidence-supported table. |
| C83 | 33 | 9.1–9.2 | Discussion organized around main findings/contributions and interpretation of improved scores. |
| C84 | 36 | 10–11 | Conclusion precedes limitations and future research. |

## Evidence limits retained

- Gemini’s configured transport-error retry maximum is absent from the archive; observed calls and the documented format policy are reported separately.
- Child ablation changes child evidence and child-specific instructions, so it is not described as an evidence-only causal test.
- The complete label audit covers all outputs; the external semantic review covers 108 purposively selected outputs, with no independent rater or population-wide hallucination-rate claim.
- Exact gold matches are not automatically treated as externally verified facts. Supported non-gold additions are distinguished from contradictions and unresolved assertions.