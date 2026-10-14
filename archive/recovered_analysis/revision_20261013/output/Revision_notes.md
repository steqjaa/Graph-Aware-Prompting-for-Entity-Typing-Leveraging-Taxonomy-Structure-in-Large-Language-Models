# Revision notes

Revised on 13 October 2026. All numerical inference results were recomputed from the four newly supplied result archives. The older PDF supplies the review requests, not numerical results.

## Main changes

- Rebuilt the result tables and figures for Gemini and Llama seeds 0, 42 and 123.
- Reported unsuccessful outputs and paired denominators, seed variability and failure sensitivity.
- Rewrote formal preliminaries, strengthened related work and expanded methods, prompt and retry documentation.
- Added complete prompts and output-inspection examples.
- Reframed the final paragraph of §7.5 as the experimental scope.

The updated child-removal effects are −0.0024 (Direct), −0.0108 (CoT) and −0.0134 (Contrastive), each with a 95% paired interval containing zero. The draft therefore does not retain the previous claim of a reliable harmful contrastive child effect.

## Response to each PDF comment

1. **Structure and headings** (reviewed PDF page 1). Numbered chapters and subsections, a page-numbered contents page and navigation bookmarks.

2. **Title wording** (reviewed PDF page 1). Graph-Aware is used consistently in the title.

3. **Graph treatment definition** (reviewed PDF page 2). Abstract; Definitions 2.9–2.10; §§5.3, 7.3 and 8.1 distinguish evidence from added instructions.

4. **Gemini identity** (reviewed PDF page 3). §5.4 gives stable model ID, Gemini Developer API, UTC dates, provider model documentation and model card.

5. **Introduction subsections and outline** (reviewed PDF page 4). §§1.1–1.3 provide numbered subsections and a thesis outline.

6. **Running example and lexical cue** (reviewed PDF page 4). Framboid replaces the literal-label example; §§4.2 and 7.6 quantify 265 label-present / 735 absent cases and their paired gains.

7. **Scientific contributions** (reviewed PDF page 4). C1–C3 are in §1.3; condensed development history is in §5.6.

8. **Research questions** (reviewed PDF page 5). RQ1–RQ4 are explicit in §1.2 and linked to results, discussion and conclusion.

9. **Hypotheses** (reviewed PDF page 5). H1–H2 are stated in §1.2 and assessed in §§7.3 and 7.6.

10. **Formal preliminaries** (reviewed PDF page 6). Chapter 2 rewritten around numbered definitions, with scoring definitions in Chapter 6.

11. **Depth versus centre** (reviewed PDF page 6). Definitions 2.3 and 2.5 separate shortest depth, graph-minimal centre and deepest-gold target.

12. **Requested definition inventory** (reviewed PDF page 6). Definitions 2.1–2.10 cover every listed concept, including entity, gold, normalization and condition.

13. **Gold closure** (reviewed PDF page 7). §4.2 reports 16 non-closed gold sets; §6.3 defines ancestor augmentation and §7.7 distinguishes extra ancestors.

14. **Move methods and results out of preliminaries** (reviewed PDF page 7). Chapter 2 contains definitions; prompts and decoding are in Chapter 5, scoring in Chapter 6 and outcomes in Chapter 7.

15. **In-context wording** (reviewed PDF page 8). In-context demonstrations terminology used consistently.

16. **One-shot terminology** (reviewed PDF page 8). §5.1 explicitly identifies the single external worked example.

17. **External demonstration** (reviewed PDF page 8). All 37,353 response metadata records use Rod Fergusson, outside the 1,000-entity cohort; every prompt contains the demonstration.

18. **CoT terminology** (reviewed PDF page 9). Chain-of-thought terminology is standardized.

19. **Unfaithful explanations citation** (reviewed PDF page 9). Turpin et al. (2023) is cited in §3.3; explanations are not treated as verified internal traces.

20. **Expand related work** (reviewed PDF page 11). Chapter 3 and Table 3.1 add Xu & Barbosa, Chen et al., Onoe et al., ConnectE, TET, LLM-based typing and Wikidata.

21. **Positive research gap** (reviewed PDF page 12). The final paragraph of Chapter 3 states the matched local-context gap positively.

22. **Dataset and context statistics** (reviewed PDF page 13). Tables 4.1–4.2 give graph size, gold size, emptiness and truncation for all relations.

23. **Centre tie-break and target discrepancy** (reviewed PDF page 14). Definition 2.5 states string ordering; Table 4.1 reports 136 multiple-deepest cases and 89 centres outside Gmax.

24. **Reference-type terminology** (reviewed PDF page 14). Centre label is defined and used; reference_type appears only as an archive-field term.

25. **Section numbering** (reviewed PDF page 16). Dataset subsections are 4.1–4.4.

26. **Modal depth correction** (reviewed PDF page 16). §4.4 identifies depth 7 as the mode and depths 6–7 as 46.5% combined.

27. **Method order, pipeline, algorithm, prompts** (reviewed PDF page 17). Chapter 5 begins with the final pipeline and extraction algorithm; Appendix B includes all nine exact prompts.

28. **Instruction effect control** (reviewed PDF page 19). §§5.3 and 7.3 inspect the exact changes; Table 7.4 gives childless comparisons. The archive shows both graph-note and formatting changes, so this is correctly described as a partial control.

29. **Gemini thinking and limits** (reviewed PDF page 20). §5.4 reports omitted thinking override, provider default documentation and 65,536 output limit; Table 5.2 reports thought tokens.

30. **Llama revision and runtime** (reviewed PDF page 20). §5.4 gives the full checkpoint hash and recorded environment; official vLLM release is referenced.

31. **Temperature and seeds** (reviewed PDF page 20). §5.4 explains T=1.0 and fixed seeds; Table 7.5 gives common-cohort means and observed across-seed SD.

32. **Retry policy and validity** (reviewed PDF page 21). §5.5 and Tables 5.3–5.4 report retry triggers, limits, first validity and condition-specific extra requests.

33. **Gemini recovery and fresh-run naming** (reviewed PDF page 21). §5.5 explains the fresh external-prompt result set and includes Gemini attempts; earlier results are excluded.

34. **Worked metric example** (reviewed PDF page 22). §6.3 illustrates precision, recall, F1, RR, nDCG and hierarchical F1.

35. **Figure cross-references** (reviewed PDF page 26). Results prose explicitly refers to Figures 7.1–7.4; Figure 5.1 is introduced in methods.

36. **Merge overlapping F1/P/R sections** (reviewed PDF page 27). §7.1 combines F1 with precision and recall.

37. **Generated type counts** (reviewed PDF page 27). Figure 7.4 and supplementary scores report distinct and raw list lengths.

38. **Interpret results** (reviewed PDF page 29). §§7.1–7.4 discuss vocabulary alignment, hierarchy recovery and model/style interactions as interpretations.

39. **Motivation for child ablation** (reviewed PDF page 30). §7.5 begins with the specificity-versus-distraction rationale and eligible cohort design.

40. **Figure caption source wording** (reviewed PDF page 30). Redundant source wording is removed from all captions.

41. **Direct versus standard** (reviewed PDF page 30). Direct is the manuscript term; internal standard names remain in reproducibility tables and exact prompts.

42. **Interpret depth results** (reviewed PDF page 31). §7.6 assesses H2 and explains branch, gold-length and lexical-cue alternatives.

43. **Examples table** (reviewed PDF page 32). Table 7.12 uses actual new Gemini and Llama outputs with qualified interpretations.

44. **Discussion subheadings** (reviewed PDF page 33). Chapter 8 has four explicit findings-oriented subsections.

45. **Conclusion before future work** (reviewed PDF page 36). Chapter 9 concludes; experimental boundaries and future research follow in Chapter 10.

## Additional scope feedback

The last paragraph of §7.5 describes the model, seed, prompt styles, eligible cohort and relation to the wider experimental design positively. It no longer frames the absence of a Gemini child-removal run as that paragraph’s limitation.

## Evidence qualifications

The graph contrasts change evidence and instructions. The child-removal templates also alter child-specific wording. Non-gold labels are not automatically hallucinations. Qualitative case judgments are illustrative and are not an independently annotated hallucination rate. The AI-assistance statement describes the substantive work performed on this revision and requires author review before submission.

## Reproduction

Run recompute.py, audit_extra.py, build_revision.py and finalize_pdf.py in that order, with the original input archives at the paths described in Appendix C. The package includes the original scorer used by the re-analysis, the new scripts and all derived numerical tables. The large original result archives and label file are not duplicated. No API calls are needed for the re-analysis.
