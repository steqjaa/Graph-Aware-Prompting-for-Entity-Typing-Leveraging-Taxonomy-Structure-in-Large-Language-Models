#!/usr/bin/env python3
"""
Thesis-focused plotting for Graph-Aware Prompting for Entity Typing.

This script reuses existing scored model outputs. It does NOT call the LLM.
It deliberately uses the original experiment terminology:
  - Standard prompt
  - Standard + taxonomy
  - CoT prompt
  - CoT + taxonomy
  - Contrastive CoT prompt
  - Contrastive CoT + taxonomy

All outputs go into <RESULTS>/plot_design_v3/ so they are easy to remove.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean


METHODS = [
    ("Standard prompt", "standard", "graph_aware_standard"),
    ("CoT prompt", "cot_no_graph", "cot_graph"),
    ("Contrastive CoT prompt", "contrastive_cot_no_graph", "contrastive_cot_graph"),
]

CONDITION_LABELS = {
    "standard": "Standard",
    "graph_aware_standard": "Standard + taxonomy",
    "cot_no_graph": "CoT",
    "cot_graph": "CoT + taxonomy",
    "contrastive_cot_no_graph": "Contrastive CoT",
    "contrastive_cot_graph": "Contrastive CoT + taxonomy",
}

# This is intentionally named "predefined generic labels", not "informative types".
# It reproduces the secondary analysis already used by score_ranked_results.py.
GENERIC_LABELS = {
    "thing",
    "entity",
    "object",
    "person",
    "person or organization",
    "corporate body",
}

BUCKETS = ["0%", "1–25%", "26–50%", "51–75%", "76–99%", "100%"]

RELATION_ORDER = [
    "gold",
    "broader_related_extra",
    "narrower_related_extra",
    "sibling_related_extra",
    "other_wikc_extra",
    "out_of_taxonomy",
]

RELATION_LABELS = {
    "gold": "Exact gold",
    "broader_related_extra": "Broader than gold",
    "narrower_related_extra": "Narrower than gold",
    "sibling_related_extra": "Sibling of gold",
    "other_wikc_extra": "Other WiKC",
    "out_of_taxonomy": "Unmapped / out of taxonomy",
}


def normalize_label(text):
    text = (text or "").casefold().strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def jlist(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def num(value):
    if value in (None, ""):
        return None
    return float(value)


def avg(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return mean(vals) if vals else None


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bucket_for(value):
    value = float(value)
    if value == 0:
        return "0%"
    if value == 1:
        return "100%"
    if value <= 0.25:
        return "1–25%"
    if value <= 0.50:
        return "26–50%"
    if value <= 0.75:
        return "51–75%"
    return "76–99%"


def condition_index(detailed):
    idx = defaultdict(dict)
    for row in detailed:
        idx[(row["model"], row["entity_id"])][row["condition"]] = row
    return idx


def build_main_summary(detailed):
    idx = condition_index(detailed)
    grouped = defaultdict(list)

    for (model, entity_id), conditions in idx.items():
        for method, no_cond, graph_cond in METHODS:
            a = conditions.get(no_cond)
            b = conditions.get(graph_cond)
            if a is not None and b is not None:
                grouped[(model, method)].append((a, b))

    rows = []
    for (model, method), pairs in sorted(grouped.items()):
        rows.append({
            "model": model,
            "methodology": method,
            "entities": len(pairs),
            "gold_coverage_without_taxonomy": avg([num(a["gold_coverage"]) for a, _ in pairs]),
            "gold_coverage_with_taxonomy": avg([num(b["gold_coverage"]) for _, b in pairs]),
            "gold_coverage_delta": avg([num(b["gold_coverage"]) - num(a["gold_coverage"]) for a, b in pairs]),
            "coverage_excluding_generic_without_taxonomy": avg([
                num(a.get("informative_gold_coverage")) for a, _ in pairs
            ]),
            "coverage_excluding_generic_with_taxonomy": avg([
                num(b.get("informative_gold_coverage")) for _, b in pairs
            ]),
            "coverage_excluding_generic_delta": avg([
                num(b.get("informative_gold_coverage")) - num(a.get("informative_gold_coverage"))
                for a, b in pairs
                if num(a.get("informative_gold_coverage")) is not None
                and num(b.get("informative_gold_coverage")) is not None
            ]),
            "ndcg_without_taxonomy": avg([num(a["ndcg_all"]) for a, _ in pairs]),
            "ndcg_with_taxonomy": avg([num(b["ndcg_all"]) for _, b in pairs]),
            "ndcg_delta": avg([num(b["ndcg_all"]) - num(a["ndcg_all"]) for a, b in pairs]),
        })
    return rows


def build_distribution(detailed):
    allowed = set(CONDITION_LABELS)
    counts = defaultdict(Counter)
    totals = Counter()

    for row in detailed:
        cond = row["condition"]
        if cond not in allowed:
            continue
        counts[cond][bucket_for(row["gold_coverage"])] += 1
        totals[cond] += 1

    rows = []
    for _, no_cond, graph_cond in METHODS:
        for cond in (no_cond, graph_cond):
            for bucket in BUCKETS:
                rows.append({
                    "condition": cond,
                    "condition_label": CONDITION_LABELS[cond],
                    "coverage_bucket": bucket,
                    "entity_count": counts[cond][bucket],
                    "entity_share": counts[cond][bucket] / totals[cond] if totals[cond] else 0,
                })
    return rows


def build_entity_gain_loss(detailed):
    idx = condition_index(detailed)
    rows = []

    for (model, entity_id), conditions in sorted(idx.items()):
        out = {"model": model, "entity_id": entity_id, "entity_label": None}
        total_changed = 0
        total_net = 0

        for method, no_cond, graph_cond in METHODS:
            a = conditions.get(no_cond)
            b = conditions.get(graph_cond)
            if a is None or b is None:
                continue
            out["entity_label"] = a["entity_label"]
            no_found = set(jlist(a["matched_gold_json"]))
            graph_found = set(jlist(b["matched_gold_json"]))
            gained = len(graph_found - no_found)
            lost = len(no_found - graph_found)
            out[f"{method}__gained"] = gained
            out[f"{method}__lost"] = lost
            out[f"{method}__net"] = gained - lost
            total_changed += gained + lost
            total_net += gained - lost

        if out["entity_label"] is not None:
            out["total_changed"] = total_changed
            out["total_net"] = total_net
            rows.append(out)

    rows.sort(key=lambda r: (r["total_net"], r["total_changed"]), reverse=True)
    return rows


def build_type_gain_loss(detailed):
    idx = condition_index(detailed)
    stats = defaultdict(lambda: {"gained": 0, "lost": 0, "opportunities": 0})

    for _, conditions in idx.items():
        for method, no_cond, graph_cond in METHODS:
            a = conditions.get(no_cond)
            b = conditions.get(graph_cond)
            if a is None or b is None:
                continue

            no_found = set(jlist(a["matched_gold_json"]))
            graph_found = set(jlist(b["matched_gold_json"]))
            for item in jlist(a["gold_types_json"]):
                label_norm = normalize_label(item["label"])
                key = (item["label"], item["class_id"], int(item["depth"]), method)
                stats[key]["opportunities"] += 1
                if label_norm not in no_found and label_norm in graph_found:
                    stats[key]["gained"] += 1
                elif label_norm in no_found and label_norm not in graph_found:
                    stats[key]["lost"] += 1

    # Pivot to one row per label/class/depth so plot can have three methodology columns.
    pivot = defaultdict(dict)
    for (label, class_id, depth, method), st in stats.items():
        key = (label, class_id, depth)
        pivot[key][method] = st

    rows = []
    for (label, class_id, depth), methods in pivot.items():
        row = {
            "gold_label": label,
            "gold_class_id": class_id,
            "gold_depth": depth,
            "is_predefined_generic": int(normalize_label(label) in GENERIC_LABELS),
        }
        total_changed = 0
        total_net = 0
        for method, _, _ in METHODS:
            st = methods.get(method, {"gained": 0, "lost": 0, "opportunities": 0})
            row[f"{method}__gained"] = st["gained"]
            row[f"{method}__lost"] = st["lost"]
            row[f"{method}__net"] = st["gained"] - st["lost"]
            row[f"{method}__opportunities"] = st["opportunities"]
            total_changed += st["gained"] + st["lost"]
            total_net += st["gained"] - st["lost"]
        row["total_changed"] = total_changed
        row["total_net"] = total_net
        rows.append(row)

    rows.sort(key=lambda r: (r["total_changed"], abs(r["total_net"])), reverse=True)
    return rows


def build_relation_profile(prediction_rows):
    allowed = set(CONDITION_LABELS)
    counts = defaultdict(Counter)
    totals = Counter()

    for row in prediction_rows:
        cond = row["condition"]
        if cond not in allowed:
            continue
        counts[cond][row["category"]] += 1
        totals[cond] += 1

    rows = []
    for _, no_cond, graph_cond in METHODS:
        for cond in (no_cond, graph_cond):
            total = totals[cond]
            for category in RELATION_ORDER:
                count = counts[cond][category]
                rows.append({
                    "condition": cond,
                    "condition_label": CONDITION_LABELS[cond],
                    "category": category,
                    "category_label": RELATION_LABELS[category],
                    "count": count,
                    "share": count / total if total else 0,
                })
    return rows


def build_depth_recall(detailed):
    """Recall by relative gold depth, then Graph-minus-No-Graph delta.

    relative depth 0 = most-specific gold type(s) for that entity,
    1 = one gold-depth level broader, etc. This does NOT identify parent/sibling/child
    prompt contributions; it only shows which specificity levels benefit.
    """
    idx = condition_index(detailed)
    agg = defaultdict(lambda: {"no_hits": 0, "graph_hits": 0, "n": 0})

    for _, conditions in idx.items():
        for method, no_cond, graph_cond in METHODS:
            a = conditions.get(no_cond)
            b = conditions.get(graph_cond)
            if a is None or b is None:
                continue
            no_found = set(jlist(a["matched_gold_json"]))
            graph_found = set(jlist(b["matched_gold_json"]))
            gold = jlist(a["gold_types_json"])
            max_depth = max((int(x["depth"]) for x in gold), default=0)

            for item in gold:
                rel = max_depth - int(item["depth"])
                label = normalize_label(item["label"])
                st = agg[(method, rel)]
                st["n"] += 1
                st["no_hits"] += int(label in no_found)
                st["graph_hits"] += int(label in graph_found)

    rows = []
    for (method, rel), st in sorted(agg.items()):
        no_recall = st["no_hits"] / st["n"] if st["n"] else 0
        graph_recall = st["graph_hits"] / st["n"] if st["n"] else 0
        rows.append({
            "methodology": method,
            "distance_from_most_specific_gold_type": rel,
            "gold_type_instances": st["n"],
            "recall_without_taxonomy": no_recall,
            "recall_with_taxonomy": graph_recall,
            "recall_change_percentage_points": (graph_recall - no_recall) * 100,
        })
    return rows


def plot_grouped_bars(summary, no_key, graph_key, title, ylabel, filename, out_dir, percentage=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    order = [m[0] for m in METHODS]
    lookup = {r["methodology"]: r for r in summary}
    x = np.arange(len(order))
    width = 0.34
    no_vals = [float(lookup[m][no_key]) for m in order]
    graph_vals = [float(lookup[m][graph_key]) for m in order]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    b1 = ax.bar(x - width/2, no_vals, width, label="Without taxonomy context")
    b2 = ax.bar(x + width/2, graph_vals, width, label="With taxonomy context")
    ax.set_xticks(x, order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left")

    if percentage:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            label = f"{h*100:.1f}%" if percentage else f"{h:.3f}"
            ax.text(bar.get_x()+bar.get_width()/2, h+0.018, label,
                    ha="center", va="bottom", fontsize=9)

    for i, method in enumerate(order):
        delta = graph_vals[i] - no_vals[i]
        ax.text(i, 0.965, f"Δ {delta*100:+.1f} pp" if percentage else f"Δ {delta:+.3f}",
                ha="center", va="top", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=240)
    plt.close(fig)


def plot_radial_distribution(distribution, out_dir):
    """One radial bar chart. Bar HEIGHT = entity count in coverage bucket."""
    import matplotlib.pyplot as plt
    import numpy as np

    condition_order = [
        "standard", "graph_aware_standard",
        "cot_no_graph", "cot_graph",
        "contrastive_cot_no_graph", "contrastive_cot_graph",
    ]
    lookup = defaultdict(dict)
    for row in distribution:
        lookup[row["condition"]][row["coverage_bucket"]] = int(row["entity_count"])

    bars_per_group = len(BUCKETS)
    group_gap = 1.6
    positions = []
    heights = []
    bucket_ids = []
    group_centers = []

    cursor = 0.0
    for cond in condition_order:
        start = cursor
        for bucket_i, bucket in enumerate(BUCKETS):
            positions.append(cursor)
            heights.append(lookup[cond].get(bucket, 0))
            bucket_ids.append(bucket_i)
            cursor += 1.0
        group_centers.append((start + cursor - 1.0) / 2)
        cursor += group_gap

    total_span = cursor
    angles = np.array(positions) / total_span * 2*np.pi
    width = (2*np.pi / total_span) * 0.78

    fig = plt.figure(figsize=(11.2, 10.5))
    ax = fig.add_subplot(111, projection="polar")

    # Use the default matplotlib color cycle, one color per coverage bucket.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [cycle[i % len(cycle)] for i in bucket_ids]
    bars = ax.bar(angles, heights, width=width, bottom=0, color=colors, alpha=0.9)

    for bar, h in zip(bars, heights):
        if h > 0:
            angle = bar.get_x() + bar.get_width()/2
            ax.text(angle, h + 0.35, str(h), ha="center", va="center", fontsize=7)

    center_angles = np.array(group_centers) / total_span * 2*np.pi
    ax.set_xticks(center_angles)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in condition_order], fontsize=9)
    max_h = max(heights + [1])
    ax.set_ylim(0, max_h + 2)
    ax.set_yticks(range(0, max_h + 1, max(1, math.ceil(max_h/4))))
    ax.set_ylabel("Number of entities", labelpad=28)
    ax.set_title(
        "Distribution of gold-type coverage\n"
        "Bar height = number of entities in each coverage range",
        pad=30,
    )

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=cycle[i % len(cycle)], label=b) for i, b in enumerate(BUCKETS)]
    ax.legend(handles=legend_handles, title="Gold types recovered per entity",
              bbox_to_anchor=(1.18, 1.05), loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "03_gold_coverage_distribution_radial.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_gain_loss_matrix(rows, row_label_key, title, filename, out_dir, top_n=None, mark_generic=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    if top_n is not None:
        rows = rows[:top_n]
    if not rows:
        return

    methods = [m[0] for m in METHODS]
    labels = []
    matrix = []
    annotations = []

    for row in rows:
        label = str(row[row_label_key])
        if mark_generic and int(row.get("is_predefined_generic", 0)):
            label += " *"
        labels.append(label)
        matrix.append([float(row.get(f"{m}__net", 0)) for m in methods])
        annotations.append([
            f"+{int(row.get(f'{m}__gained', 0))} / −{int(row.get(f'{m}__lost', 0))}"
            for m in methods
        ])

    data = np.array(matrix, dtype=float)
    vmax = max(abs(data.min()), abs(data.max()), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.42*len(labels))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu", norm=norm)
    ax.set_xticks(range(len(methods)), methods)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(methods)):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=7.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net change = gold types gained − gold types lost")
    ax.text(
        0.0, -0.07,
        "Cell text = +gained / −lost. Color shows the net change.",
        transform=ax.transAxes, fontsize=8, va="top"
    )
    if mark_generic:
        ax.text(
            0.0, -0.105,
            "* Predefined generic label used only in the secondary filtered analysis.",
            transform=ax.transAxes, fontsize=8, va="top"
        )
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_relation_profile(rows, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    order = [
        "standard", "graph_aware_standard",
        "cot_no_graph", "cot_graph",
        "contrastive_cot_no_graph", "contrastive_cot_graph",
    ]
    lookup = defaultdict(dict)
    for row in rows:
        lookup[row["condition"]][row["category"]] = float(row["share"])

    labels = [CONDITION_LABELS[c] for c in order]
    y = list(range(len(order)))
    left = [0.0] * len(order)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for category in RELATION_ORDER:
        vals = [lookup[c].get(category, 0.0) for c in order]
        bars = ax.barh(y, vals, left=left, label=RELATION_LABELS[category])
        # Label only segments large enough to read.
        for yi, (bar, v, lft) in enumerate(zip(bars, vals, left)):
            if v >= 0.07:
                ax.text(lft + v/2, yi, f"{v*100:.0f}%", ha="center", va="center", fontsize=7)
        left = [a+b for a, b in zip(left, vals)]

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Share of all predictions")
    ax.set_title("Prediction outcomes: exact, structurally related errors, and unmapped labels")
    ax.legend(title="Relation to gold type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_dir / "06_prediction_outcome_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_depth_delta(rows, out_dir):
    import matplotlib.pyplot as plt

    methods = [m[0] for m in METHODS]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    for method in methods:
        subset = sorted(
            [r for r in rows if r["methodology"] == method],
            key=lambda r: int(r["distance_from_most_specific_gold_type"]),
        )
        if not subset:
            continue
        x = [int(r["distance_from_most_specific_gold_type"]) for r in subset]
        y = [float(r["recall_change_percentage_points"]) for r in subset]
        ax.plot(x, y, marker="o", label=method)

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Gold-type level relative to the entity's most-specific gold type\n0 = most specific; larger numbers = broader gold types")
    ax.set_ylabel("Change in recall after adding taxonomy context (percentage points)")
    ax.set_title("Where in the gold hierarchy does taxonomy context help?")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_recall_change_by_gold_specificity.png", dpi=240)
    plt.close(fig)


def write_guide(out_dir):
    generic = ", ".join(sorted(GENERIC_LABELS))
    text = f"""# How to read the redesigned thesis plots

The plots intentionally use the experiment's original names. There is no
"Direct" condition in these figures.

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

`{generic}`

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

Only after that run should we plot parent vs sibling vs child effects.
"""
    (out_dir / "READ_ME_FIRST.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create readable thesis plots from scored results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--top-types", type=int, default=18)
    args = parser.parse_args()

    detailed_path = args.results / "weighted_detailed_scores.csv"
    prediction_path = args.results / "prediction_level_analysis.csv"
    if not detailed_path.exists():
        raise FileNotFoundError(detailed_path)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    detailed = read_csv(detailed_path)
    predictions = read_csv(prediction_path)

    out_dir = args.results / "plot_design_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_main_summary(detailed)
    distribution = build_distribution(detailed)
    entity_rows = build_entity_gain_loss(detailed)
    type_rows = build_type_gain_loss(detailed)
    relation_rows = build_relation_profile(predictions)
    depth_rows = build_depth_recall(detailed)

    write_csv(out_dir / "01_02_main_effect_summary.csv", summary)
    write_csv(out_dir / "03_gold_coverage_distribution.csv", distribution)
    write_csv(out_dir / "04_entity_gold_type_gain_loss.csv", entity_rows)
    write_csv(out_dir / "05_gold_type_gain_loss.csv", type_rows)
    write_csv(out_dir / "06_prediction_outcome_profile.csv", relation_rows)
    write_csv(out_dir / "07_recall_by_gold_specificity.csv", depth_rows)

    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        print(f"Plot dependency missing: {exc}")
        print("CSVs were written, but PNGs were skipped.")
        return

    plot_grouped_bars(
        summary,
        "gold_coverage_without_taxonomy",
        "gold_coverage_with_taxonomy",
        "Does taxonomy context improve entity typing?",
        "Mean gold-type coverage",
        "01_main_effect_overall_gold_coverage.png",
        out_dir,
        percentage=True,
    )

    plot_grouped_bars(
        summary,
        "coverage_excluding_generic_without_taxonomy",
        "coverage_excluding_generic_with_taxonomy",
        "Gold coverage after excluding predefined generic labels",
        "Mean gold-type coverage (generic labels excluded)",
        "02_main_effect_excluding_generic_labels.png",
        out_dir,
        percentage=True,
    )

    plot_radial_distribution(distribution, out_dir)
    plot_gain_loss_matrix(
        entity_rows,
        "entity_label",
        "Which entities gain or lose gold types when taxonomy context is added?",
        "04_entity_gold_type_gain_loss_matrix.png",
        out_dir,
    )
    plot_gain_loss_matrix(
        type_rows,
        "gold_label",
        "Which gold entity types are gained or lost with taxonomy context?",
        "05_gold_type_gain_loss_matrix.png",
        out_dir,
        top_n=args.top_types,
        mark_generic=True,
    )
    plot_relation_profile(relation_rows, out_dir)
    plot_depth_delta(depth_rows, out_dir)
    write_guide(out_dir)

    print("\nPlot redesign complete.")
    print("Outputs:", out_dir)
    print("\nOpen READ_ME_FIRST.md, then the PNGs in numerical order.")


if __name__ == "__main__":
    main()