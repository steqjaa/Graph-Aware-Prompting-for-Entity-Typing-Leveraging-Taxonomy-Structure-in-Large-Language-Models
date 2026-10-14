# How to read the thesis plots

- **Standard** = `standard`
- **Standard + taxonomy** = `graph_aware_standard`
- **CoT** = `cot_no_graph`
- **CoT + taxonomy** = `cot_graph`
- **Contrastive CoT** = `contrastive_cot_no_graph`
- **Contrastive CoT + taxonomy** = `contrastive_cot_graph`

## 01 — Overall gold coverage
Familiar grouped bars on a common scale. The y-axis is the percentage of all
gold types recovered, averaged over entities.

## 02 — Gold coverage excluding predefined generic labels
This is a secondary analysis, not a new notion learned from the taxonomy.
The hard-coded excluded labels are exactly:

`corporate body, entity, object, person, person or organization, thing`

The plot title says this explicitly instead of calling the result simply
"informative".

## 03 — Radial gold-coverage distribution
This is ONE radial bar chart. Bars no longer all have the same height.
Each bar corresponds to a coverage bucket (0%, 1–25%, ..., 100%), and the
bar height is the NUMBER OF ENTITIES in that bucket. The bars are grouped
around the circle by the six actual prompt conditions.

Because radial layouts make precise magnitude comparison harder, this figure
is for distribution overview. Use Figures 01–02 for exact quantitative
comparisons.

## 04 — Entity gain/loss matrix
Each cell literally says `+gained / −lost`, where the numbers are COUNTS OF
GOLD TYPES. For example, `+3 / −1` means adding taxonomy caused three gold
types to appear and one previously found gold type to disappear for that
entity under that prompting methodology. Color is the net change (3-1=+2).

## 05 — Gold-type gain/loss matrix
Same encoding as Figure 04, but rows are actual gold type labels. This answers
which entity types are specifically gained or lost. A `*` marks one of the
predefined generic labels above; these are shown rather than silently removed.

## 06 — Prediction outcome profile
Each horizontal bar sums to 100% of the predictions from one condition. It
separates exact gold predictions from broader, narrower, sibling, other WiKC,
and unmapped/out-of-taxonomy predictions. This is the plot to discuss
narrowing/over-specialization and the hallucination proxy.

`Unmapped / out of taxonomy` is not automatically a hallucination; it means
the normalized predicted label could not be mapped to a WiKC class label.

## 07 — Recall change by gold specificity
This does NOT claim parent/sibling/child causal effects. It asks a different
question: does taxonomy context help most-specific gold types more than broad
gold types? 0 means the entity's most-specific gold type; larger values are
broader gold levels.

## Parent vs sibling vs child
The current model run cannot identify which supplied relation caused the
improvement because the graph-aware prompt provided parent, sibling, and child
context together. A separate ablation run is required:

1. Standard (no taxonomy)
2. Parent-only taxonomy context
3. Sibling-only taxonomy context
4. Child-only taxonomy context
5. Full taxonomy context