Entity typing assigns semantic categories to entities. When categories form a taxonomy, a
useful prediction must identify a suitable specific type, recover relevant broader types, and
distinguish nearby alternatives. This thesis investigates whether presenting local taxonomy
structure together with instructions for interpreting it improves that task for large language
models. The central comparison holds the entity, description, and prompting style fixed
while changing both structural evidence and graph-specific instructions. Three styles are
examined: standard prompting, chain-of-thought prompting, and contrastive chain-of-thought
prompting.
We use the released 1,000-entity WiKC evaluation sample and the refined Wikidata tax-
onomy introduced by Peng, Bonald, and Alam (2024). Nine conditions are evaluated with
Gemini 3.8 Flash and Llama 3.1 8B Instruct; Llama is run with seeds 0, 42, and 123. The
final product repository of this thesis contains 9,000 completed Gemini responses and 8,991,
8,971, and 8,986 completed main Llama responses, respectively. A further child-removal
experiment at Llama seed 0 completes 1,352 of 1,353 planned responses for 451 eligible
entities. Within-model comparisons use matched successful responses. Context is selected
around an annotated gold class, whose label is omitted. Therefore, the experiment measures
the value of oracle structural evidence and not simply autonomous retrieval.
Graph-aware prompting improves mean entity F1 scores in every model-by-style compari-
son, with paired gains from 0.0451 to 0.1348. Parent-only context outperforms sibling-only
context on aggregate F1. The three Llama seeds permit a direct check of sampling varia-
tion, and the matched child study explores child-context effects. The results are interpreted
together with output length, ranking metrics, and condition-specific completion rates.
For entities with no child classes, the additional comparison uses the same entity infor-
mation and worked example, but adds an empty child-context section and instructions for
interpreting it. This empty-child control gives small positive point estimates (about +0.005
to +0.024). Gemini’s roughly +0.02 contrasts with its +0.0625 standard graph gain. This
comparison assesses whether changes to the prompt alone can affect performance when no
child labels are supplied, and whether missing child class information greatly impacts the
results or not.
These findings support graph-aware prompting as a useful source of evidence under
controlled context selection. The contribution is a reproducible prompting and evaluation
framework, together with an empirical account of where local structure helps and where it
misleads. Future work should evaluate retrieved context, independently annotated gold sets,
and controlled structural noise.
Keywords: entity typing, knowledge graphs, taxonomy, prompting, WiKC, evaluation
