from pathlib import Path
import argparse
import csv
import json
import math
import re
from collections import defaultdict

import networkx as nx


LEGACY_K = 10
COVERAGE_BUCKETS = [
    (0.0, 0.0, "0%"),
    (0.0, 0.25, "1-25%"),
    (0.25, 0.50, "26-50%"),
    (0.50, 0.75, "51-75%"),
    (0.75, 1.0, "76-99%"),
    (1.0, 1.0, "100%"),
]

# Secondary analysis requested by the supervisors: remove extremely broad
# labels so improvements on specific typing are not hidden by easy matches.
# Primary metrics still use the complete gold set.
GENERIC_ANALYSIS_TYPES = {
    "thing",
    "entity",
    "object",
    "person",
    "person or organization",
    "corporate body",
}

# Controlled comparisons. Each pair keeps the reasoning strategy fixed and
# changes only the source/availability of taxonomy context. Conditions that
# are absent from a result folder are simply skipped.
PAIRWISE_COMPARISONS = [
    ("direct_oracle_taxonomy", "standard", "graph_aware_standard", "oracle"),
    ("cot_oracle_taxonomy", "cot_no_graph", "cot_graph", "oracle"),
    (
        "contrastive_oracle_taxonomy",
        "contrastive_cot_no_graph",
        "contrastive_cot_graph",
        "oracle",
    ),
    ("direct_shuffled_control", "standard", "shuffled_graph_standard", "shuffled"),
    ("direct_retrieved_taxonomy", "standard", "retrieved_graph_standard", "retrieved"),
    ("cot_retrieved_taxonomy", "cot_no_graph", "cot_retrieved_graph", "retrieved"),
    (
        "contrastive_retrieved_taxonomy",
        "contrastive_cot_no_graph",
        "contrastive_cot_retrieved_graph",
        "retrieved",
    ),
    (
        "direct_oracle_vs_retrieved",
        "retrieved_graph_standard",
        "graph_aware_standard",
        "oracle_vs_retrieved",
    ),
    (
        "cot_oracle_vs_retrieved",
        "cot_retrieved_graph",
        "cot_graph",
        "oracle_vs_retrieved",
    ),
    (
        "contrastive_oracle_vs_retrieved",
        "contrastive_cot_retrieved_graph",
        "contrastive_cot_graph",
        "oracle_vs_retrieved",
    ),
]


# =========================================================
# Normalization and matching
# =========================================================

def normalize_label(text):
    text = text.casefold().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def load_aliases(alias_file):
    """Load an optional fixed alias mapping for evaluation."""
    if alias_file is None or not alias_file.exists():
        return {}

    raw = json.loads(alias_file.read_text(encoding="utf-8"))

    return {
        normalize_label(source): normalize_label(target)
        for source, target in raw.items()
    }


def canonical_prediction(label, aliases):
    normalized = normalize_label(label)
    return aliases.get(normalized, normalized)


def informative_gold_types(gold_types):
    return [
        item
        for item in gold_types
        if normalize_label(item["label"]) not in GENERIC_ANALYSIS_TYPES
    ]


def informative_predictions(predictions, aliases):
    return [
        prediction
        for prediction in predictions
        if canonical_prediction(prediction, aliases) not in GENERIC_ANALYSIS_TYPES
    ]


def mean_valid(values):
    values = [value for value in values if value not in (None, "")]
    return sum(values) / len(values) if values else None


# =========================================================
# Ranking metrics
# =========================================================

def relevance_lookup(gold_types):
    """
    Assign hierarchical relevance from WiKC depth.

    The deepest listed type receives relevance 1.0. Broader listed
    classes receive partial credit proportional to their depth.
    """
    maximum_depth = max(item["depth"] for item in gold_types)
    lookup = {}

    for item in gold_types:
        label = normalize_label(item["label"])
        relevance = (
            item["depth"] / maximum_depth
            if maximum_depth > 0
            else 1.0
        )
        lookup[label] = max(lookup.get(label, 0.0), relevance)

    most_specific_labels = {
        normalize_label(item["label"])
        for item in gold_types
        if item["depth"] == maximum_depth
    }

    return lookup, most_specific_labels


def dcg_at_k(relevances, k):
    return sum(
        (2.0 ** relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg(predictions, gold_types, aliases, k=None):
    gold_relevance, _ = relevance_lookup(gold_types)

    if k is None:
        k = max(len(predictions), len(gold_relevance), 1)

    observed_relevances = []
    matched_gold_labels = set()

    for prediction in predictions[:k]:
        canonical = canonical_prediction(prediction, aliases)
        relevance = gold_relevance.get(canonical, 0.0)

        if canonical in matched_gold_labels:
            relevance = 0.0
        elif relevance > 0:
            matched_gold_labels.add(canonical)

        observed_relevances.append(relevance)

    while len(observed_relevances) < k:
        observed_relevances.append(0.0)

    ideal_relevances = sorted(
        gold_relevance.values(),
        reverse=True,
    )[:k]

    while len(ideal_relevances) < k:
        ideal_relevances.append(0.0)

    dcg = dcg_at_k(observed_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(predictions, gold_types, aliases, k=None):
    _, most_specific_labels = relevance_lookup(gold_types)
    inspected = predictions if k is None else predictions[:k]

    for rank, prediction in enumerate(inspected, start=1):
        if canonical_prediction(prediction, aliases) in most_specific_labels:
            return 1.0 / rank

    return 0.0


def hit_at_k(predictions, gold_types, aliases, k=None):
    _, most_specific_labels = relevance_lookup(gold_types)
    inspected = predictions if k is None else predictions[:k]

    return int(
        any(
            canonical_prediction(prediction, aliases)
            in most_specific_labels
            for prediction in inspected
        )
    )


def gold_set_metrics(predictions, gold_types, aliases):
    gold_order = [normalize_label(item["label"]) for item in gold_types]
    gold_set = set(gold_order)
    predicted = [canonical_prediction(p, aliases) for p in predictions]

    matched = []
    matched_set = set()

    for label in predicted:
        if label in gold_set and label not in matched_set:
            matched.append(label)
            matched_set.add(label)

    missing = [label for label in gold_order if label not in matched_set]
    extras = [
        predictions[index]
        for index, label in enumerate(predicted)
        if label not in gold_set
    ]

    coverage = len(matched_set) / len(gold_set) if gold_set else 0.0
    precision = len(matched_set) / len(predicted) if predicted else 0.0

    ideal_rank = {label: rank for rank, label in enumerate(gold_order, start=1)}
    rank_errors = []

    for rank, label in enumerate(predicted, start=1):
        if label in ideal_rank:
            rank_errors.append(abs(rank - ideal_rank[label]))

    return {
        "gold_count": len(gold_set),
        "predictions_count": len(predicted),
        "gold_matches_count": len(matched_set),
        "gold_coverage": coverage,
        "gold_precision": precision,
        "missing_gold": missing,
        "extra_predictions": extras,
        "mean_absolute_gold_rank_error": (
            sum(rank_errors) / len(rank_errors)
            if rank_errors
            else None
        ),
    }


# =========================================================
# WiKC graph analysis for additional predictions
# =========================================================

def build_graph(file_path):
    graph = nx.DiGraph()

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) == 2:
                child, parent = row
                graph.add_edge(child, parent)

    return graph


def load_label_index(file_path):
    """Map normalized WiKC labels to one or more class IDs."""
    label_to_ids = defaultdict(set)
    id_to_label = {}

    pattern = re.compile(
        r'^(wd:Q\d+)\s+rdfs:label\s+"((?:\\.|[^"])*)"\s+\.$'
    )

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line.strip())

            if not match:
                continue

            class_id = match.group(1)
            label = match.group(2).replace(r'\"', '"')
            normalized = normalize_label(label)

            label_to_ids[normalized].add(class_id)
            id_to_label[class_id] = label

    return dict(label_to_ids), id_to_label


def relation_cache_for_gold(graph, gold_ids):
    broader = set()
    narrower = set()
    gold_parents = {}

    for gold_id in gold_ids:
        if gold_id not in graph:
            gold_parents[gold_id] = set()
            continue

        # child -> parent edges:
        # descendants(gold) are broader ancestors reachable upward;
        # ancestors(gold) are narrower descendants that can reach gold.
        broader.update(nx.descendants(graph, gold_id))
        narrower.update(nx.ancestors(graph, gold_id))
        gold_parents[gold_id] = set(graph.successors(gold_id))

    return broader, narrower, gold_parents


def classify_prediction(
    prediction,
    gold_types,
    aliases,
    graph,
    label_to_ids,
    id_to_label,
):
    canonical = canonical_prediction(prediction, aliases)
    gold_label_to_ids = defaultdict(set)
    gold_ids = set()

    for item in gold_types:
        normalized = normalize_label(item["label"])
        gold_label_to_ids[normalized].add(item["class_id"])
        gold_ids.add(item["class_id"])

    if canonical in gold_label_to_ids:
        return "gold", "exact gold label", sorted(gold_label_to_ids[canonical])

    prediction_ids = set(label_to_ids.get(canonical, set()))

    if not prediction_ids:
        return "out_of_taxonomy", "label not found in WiKC", []

    if prediction_ids & gold_ids:
        return "gold", "same WiKC class ID as a gold type", sorted(prediction_ids & gold_ids)

    broader, narrower, gold_parents = relation_cache_for_gold(graph, gold_ids)

    if prediction_ids & broader:
        ids = sorted(prediction_ids & broader)
        detail = "broader ancestor of a gold class"
        return "broader_related_extra", detail, ids

    if prediction_ids & narrower:
        ids = sorted(prediction_ids & narrower)
        detail = "narrower descendant of a gold class"
        return "narrower_related_extra", detail, ids

    sibling_matches = set()

    for prediction_id in prediction_ids:
        if prediction_id not in graph:
            continue

        prediction_parents = set(graph.successors(prediction_id))

        for gold_id, parents in gold_parents.items():
            if prediction_parents & parents:
                sibling_matches.add(prediction_id)

    if sibling_matches:
        return (
            "sibling_related_extra",
            "shares a direct parent with a gold class",
            sorted(sibling_matches),
        )

    readable = [id_to_label.get(item, item) for item in sorted(prediction_ids)]
    detail = "valid WiKC class but no ancestor/descendant/direct-sibling relation to gold"
    if readable:
        detail += f" ({'; '.join(readable[:3])})"

    return "other_wikc_extra", detail, sorted(prediction_ids)


# =========================================================
# Scoring pipeline
# =========================================================

def resolve_metadata_path(result_file, metadata_path_text):
    path = Path(metadata_path_text)

    if path.exists():
        return path

    # Typical layout: project/generated_ranked_results/model/example_x/result.json
    project_root = result_file.parents[3]
    candidate = project_root / path

    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Metadata file not found: {metadata_path_text}")


def score_result_file(
    result_file,
    aliases,
    taxonomy_resources=None,
):
    result = json.loads(result_file.read_text(encoding="utf-8"))

    if result.get("status") != "completed":
        return None, []

    metadata_path = resolve_metadata_path(
        result_file,
        result["metadata_path"],
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    predictions = result.get("predictions", [])
    gold_types = metadata["gold_types"]
    set_metrics = gold_set_metrics(predictions, gold_types, aliases)

    informative_gold = informative_gold_types(gold_types)
    informative_preds = informative_predictions(predictions, aliases)
    if informative_gold:
        informative_set_metrics = gold_set_metrics(
            informative_preds, informative_gold, aliases
        )
        informative_ndcg = ndcg(informative_preds, informative_gold, aliases)
        informative_mrr = reciprocal_rank(
            informative_preds, informative_gold, aliases
        )
        informative_hit = hit_at_k(
            informative_preds, informative_gold, aliases, None
        )
    else:
        informative_set_metrics = {
            "gold_count": 0,
            "predictions_count": len(informative_preds),
            "gold_matches_count": 0,
            "gold_coverage": None,
            "gold_precision": None,
            "missing_gold": [],
            "extra_predictions": informative_preds,
            "mean_absolute_gold_rank_error": None,
        }
        informative_ndcg = None
        informative_mrr = None
        informative_hit = None

    prediction_rows = []
    category_counts = defaultdict(int)

    for rank, prediction in enumerate(predictions, start=1):
        canonical = canonical_prediction(prediction, aliases)

        if taxonomy_resources is None:
            gold_labels = {
                normalize_label(item["label"])
                for item in gold_types
            }
            if canonical in gold_labels:
                category = "gold"
                relation = "exact gold label"
            else:
                category = "non_gold_unchecked"
                relation = "taxonomy files unavailable"
            class_ids = []
        else:
            graph, label_to_ids, id_to_label = taxonomy_resources
            category, relation, class_ids = classify_prediction(
                prediction,
                gold_types,
                aliases,
                graph,
                label_to_ids,
                id_to_label,
            )

        category_counts[category] += 1

        prediction_rows.append(
            {
                "example_number": result["example_number"],
                "entity_id": result["entity_id"],
                "entity_label": result["entity_label"],
                "condition": result["condition"],
                "model": result["model"],
                "rank": rank,
                "prediction": prediction,
                "canonical_prediction": canonical,
                "category": category,
                "relation_to_gold": relation,
                "matched_class_ids_json": json.dumps(class_ids),
            }
        )

    total_predictions = len(predictions)
    extra_count = total_predictions - category_counts["gold"]
    hallucination_count = category_counts["out_of_taxonomy"]
    taxonomy_related_extra_count = sum(
        category_counts[name]
        for name in (
            "broader_related_extra",
            "narrower_related_extra",
            "sibling_related_extra",
        )
    )

    _, most_specific_labels = relevance_lookup(gold_types)
    most_specific_rank = None

    for rank, prediction in enumerate(predictions, start=1):
        if canonical_prediction(prediction, aliases) in most_specific_labels:
            most_specific_rank = rank
            break

    detailed = {
        "example_number": result["example_number"],
        "entity_id": result["entity_id"],
        "entity_label": result["entity_label"],
        "condition": result["condition"],
        "model": result["model"],
        "reference_type": metadata["reference_type"],
        "predictions_json": json.dumps(predictions, ensure_ascii=False),
        "gold_types_json": json.dumps(gold_types, ensure_ascii=False),
        "matched_gold_json": json.dumps(
            [
                normalize_label(item["label"])
                for item in gold_types
                if normalize_label(item["label"])
                not in set_metrics["missing_gold"]
            ],
            ensure_ascii=False,
        ),
        "predictions_count": set_metrics["predictions_count"],
        "gold_type_count": set_metrics["gold_count"],
        "gold_matches_count": set_metrics["gold_matches_count"],
        "gold_coverage": set_metrics["gold_coverage"],
        "gold_precision": set_metrics["gold_precision"],
        "missing_gold_json": json.dumps(
            set_metrics["missing_gold"],
            ensure_ascii=False,
        ),
        "extra_predictions_json": json.dumps(
            set_metrics["extra_predictions"],
            ensure_ascii=False,
        ),
        "extra_prediction_count": len(set_metrics["extra_predictions"]),
        "mean_absolute_gold_rank_error": (
            set_metrics["mean_absolute_gold_rank_error"]
        ),
        "most_specific_gold_rank": most_specific_rank or "",
        "reciprocal_rank": reciprocal_rank(
            predictions,
            gold_types,
            aliases,
        ),
        "ndcg_all": ndcg(
            predictions,
            gold_types,
            aliases,
        ),
        "hit_at_1": hit_at_k(predictions, gold_types, aliases, 1),
        "hit_at_3": hit_at_k(predictions, gold_types, aliases, 3),
        "hit_at_5": hit_at_k(predictions, gold_types, aliases, 5),
        "hit_at_10": hit_at_k(predictions, gold_types, aliases, LEGACY_K),
        "hit_anywhere": hit_at_k(predictions, gold_types, aliases, None),
        # Keep the old names so earlier tables/scripts still work.
        "reciprocal_rank_at_10": reciprocal_rank(
            predictions,
            gold_types,
            aliases,
            LEGACY_K,
        ),
        "ndcg_at_10": ndcg(
            predictions,
            gold_types,
            aliases,
            LEGACY_K,
        ),
        "informative_gold_type_count": informative_set_metrics["gold_count"],
        "informative_predictions_count": informative_set_metrics["predictions_count"],
        "informative_gold_matches_count": informative_set_metrics["gold_matches_count"],
        "informative_gold_coverage": informative_set_metrics["gold_coverage"],
        "informative_gold_precision": informative_set_metrics["gold_precision"],
        "informative_ndcg": informative_ndcg,
        "informative_mrr": informative_mrr,
        "informative_hit_anywhere": informative_hit,
        "taxonomy_related_extra_count": taxonomy_related_extra_count,
        "other_wikc_extra_count": category_counts["other_wikc_extra"],
        "out_of_taxonomy_count": hallucination_count,
        "hallucination_rate": (
            hallucination_count / total_predictions
            if total_predictions
            else 0.0
        ),
        "extra_hallucination_rate": (
            hallucination_count / extra_count
            if extra_count > 0
            else 0.0
        ),
    }

    return detailed, prediction_rows


def coverage_bucket(value):
    if value == 0:
        return "0%"
    if value == 1:
        return "100%"
    if value <= 0.25:
        return "1-25%"
    if value <= 0.50:
        return "26-50%"
    if value <= 0.75:
        return "51-75%"
    return "76-99%"


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def aggregate_conditions(detailed_rows, prediction_rows, taxonomy_available):
    by_condition = defaultdict(list)
    for row in detailed_rows:
        by_condition[(row["model"], row["condition"])].append(row)

    prediction_categories = defaultdict(lambda: defaultdict(int))
    for row in prediction_rows:
        key = (row["model"], row["condition"])
        prediction_categories[key][row["category"]] += 1

    summaries = []

    for key, rows in sorted(by_condition.items()):
        model, condition = key
        prediction_total = sum(row["predictions_count"] for row in rows)
        gold_prediction_total = sum(row["gold_matches_count"] for row in rows)
        categories = prediction_categories[key]
        hallucinations = categories["out_of_taxonomy"]
        related_extras = sum(
            categories[name]
            for name in (
                "broader_related_extra",
                "narrower_related_extra",
                "sibling_related_extra",
            )
        )

        summaries.append(
            {
                "model": model,
                "condition": condition,
                "entities_scored": len(rows),
                "mean_predictions": mean(row["predictions_count"] for row in rows),
                "mean_gold_coverage": mean(row["gold_coverage"] for row in rows),
                "median_gold_coverage": median(row["gold_coverage"] for row in rows),
                "mean_gold_precision": mean(row["gold_precision"] for row in rows),
                "mean_informative_gold_coverage": mean_valid(
                    row["informative_gold_coverage"] for row in rows
                ),
                "mean_informative_gold_precision": mean_valid(
                    row["informative_gold_precision"] for row in rows
                ),
                "mean_informative_ndcg": mean_valid(
                    row["informative_ndcg"] for row in rows
                ),
                "mean_informative_mrr": mean_valid(
                    row["informative_mrr"] for row in rows
                ),
                "informative_hit_rate_anywhere": mean_valid(
                    row["informative_hit_anywhere"] for row in rows
                ),
                "mrr": mean(row["reciprocal_rank"] for row in rows),
                "mean_ndcg_all": mean(row["ndcg_all"] for row in rows),
                "hit_rate_at_1": mean(row["hit_at_1"] for row in rows),
                "hit_rate_at_3": mean(row["hit_at_3"] for row in rows),
                "hit_rate_at_5": mean(row["hit_at_5"] for row in rows),
                "hit_rate_at_10": mean(row["hit_at_10"] for row in rows),
                "hit_rate_anywhere": mean(row["hit_anywhere"] for row in rows),
                "mrr_at_10": mean(row["reciprocal_rank_at_10"] for row in rows),
                "mean_ndcg_at_10": mean(row["ndcg_at_10"] for row in rows),
                "gold_prediction_share": (
                    gold_prediction_total / prediction_total
                    if prediction_total
                    else 0.0
                ),
                "taxonomy_related_extra_share": (
                    related_extras / prediction_total
                    if taxonomy_available and prediction_total
                    else ""
                ),
                "out_of_taxonomy_hallucination_rate": (
                    hallucinations / prediction_total
                    if taxonomy_available and prediction_total
                    else ""
                ),
            }
        )

    return summaries


def build_distribution_rows(detailed_rows):
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    order = [item[2] for item in COVERAGE_BUCKETS]

    for row in detailed_rows:
        key = (row["model"], row["condition"])
        bucket = coverage_bucket(row["gold_coverage"])
        counts[key][bucket] += 1
        totals[key] += 1

    rows = []
    for key in sorted(counts):
        model, condition = key
        for bucket in order:
            count = counts[key][bucket]
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "coverage_bucket": bucket,
                    "entity_count": count,
                    "entity_share": count / totals[key] if totals[key] else 0.0,
                }
            )
    return rows


def build_hit_by_rank_rows(detailed_rows):
    by_condition = defaultdict(list)
    max_rank = 1

    for row in detailed_rows:
        key = (row["model"], row["condition"])
        rank = row["most_specific_gold_rank"]
        rank = int(rank) if rank != "" else None
        by_condition[key].append(rank)
        max_rank = max(max_rank, row["predictions_count"])

    rows = []
    for key in sorted(by_condition):
        model, condition = key
        ranks = by_condition[key]
        total = len(ranks)

        for cutoff in range(1, max_rank + 1):
            hits = sum(rank is not None and rank <= cutoff for rank in ranks)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "rank_cutoff": cutoff,
                    "hit_rate": hits / total if total else 0.0,
                }
            )

    return rows


def build_prediction_category_rows(prediction_rows):
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for row in prediction_rows:
        key = (row["model"], row["condition"])
        counts[key][row["category"]] += 1
        totals[key] += 1

    rows = []
    for key in sorted(counts):
        model, condition = key
        for category, count in sorted(counts[key].items()):
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "category": category,
                    "prediction_count": count,
                    "prediction_share": count / totals[key] if totals[key] else 0.0,
                }
            )
    return rows


def build_comparison_rows(summary_rows):
    by_model = defaultdict(dict)
    for row in summary_rows:
        by_model[row["model"]][row["condition"]] = row

    comparisons = []
    for model, conditions in sorted(by_model.items()):
        baseline = conditions.get("standard")
        if baseline is None:
            continue

        for condition, row in sorted(conditions.items()):
            if condition == "standard":
                continue

            comparisons.append(
                {
                    "model": model,
                    "condition": condition,
                    "delta_gold_coverage_vs_standard": (
                        row["mean_gold_coverage"] - baseline["mean_gold_coverage"]
                    ),
                    "delta_ndcg_all_vs_standard": (
                        row["mean_ndcg_all"] - baseline["mean_ndcg_all"]
                    ),
                    "delta_mrr_vs_standard": row["mrr"] - baseline["mrr"],
                    "delta_hit_anywhere_vs_standard": (
                        row["hit_rate_anywhere"] - baseline["hit_rate_anywhere"]
                    ),
                }
            )

    return comparisons


def _json_list(value):
    if not value:
        return []
    return json.loads(value)


def _prediction_map(row):
    predictions = _json_list(row["predictions_json"])
    return {normalize_label(item): item for item in predictions}


def _rank_value(value):
    if value in (None, ""):
        return None
    return int(value)


def build_pairwise_taxonomy_rows(detailed_rows):
    by_entity = defaultdict(dict)
    for row in detailed_rows:
        key = (row["model"], row["entity_id"])
        by_entity[key][row["condition"]] = row

    output = []
    for (model, entity_id), conditions in sorted(by_entity.items()):
        for comparison, base_condition, variant_condition, context_type in PAIRWISE_COMPARISONS:
            base = conditions.get(base_condition)
            variant = conditions.get(variant_condition)
            if base is None or variant is None:
                continue

            base_matches = set(_json_list(base["matched_gold_json"]))
            variant_matches = set(_json_list(variant["matched_gold_json"]))
            gold_types = _json_list(base["gold_types_json"])
            gold_labels = {normalize_label(item["label"]) for item in gold_types}

            gained = sorted(variant_matches - base_matches)
            lost = sorted(base_matches - variant_matches)
            shared = sorted(base_matches & variant_matches)
            neither = sorted(gold_labels - (base_matches | variant_matches))

            base_predictions = _prediction_map(base)
            variant_predictions = _prediction_map(variant)
            added_keys = [key for key in variant_predictions if key not in base_predictions]
            removed_keys = [key for key in base_predictions if key not in variant_predictions]
            shared_pred_keys = [key for key in variant_predictions if key in base_predictions]

            base_rank = _rank_value(base["most_specific_gold_rank"])
            variant_rank = _rank_value(variant["most_specific_gold_rank"])
            if base_rank is None and variant_rank is not None:
                most_specific_change = "miss_to_hit"
            elif base_rank is not None and variant_rank is None:
                most_specific_change = "hit_to_miss"
            elif base_rank is None and variant_rank is None:
                most_specific_change = "both_miss"
            elif variant_rank < base_rank:
                most_specific_change = "both_hit_rank_improved"
            elif variant_rank > base_rank:
                most_specific_change = "both_hit_rank_worsened"
            else:
                most_specific_change = "both_hit_same_rank"

            coverage_delta = variant["gold_coverage"] - base["gold_coverage"]
            if coverage_delta > 1e-12:
                coverage_outcome = "improved"
            elif coverage_delta < -1e-12:
                coverage_outcome = "harmed"
            else:
                coverage_outcome = "unchanged"

            def delta_optional(key):
                left = base.get(key)
                right = variant.get(key)
                if left in (None, "") or right in (None, ""):
                    return None
                return right - left

            output.append(
                {
                    "model": model,
                    "entity_id": entity_id,
                    "entity_label": base["entity_label"],
                    "comparison": comparison,
                    "context_type": context_type,
                    "without_condition": base_condition,
                    "with_condition": variant_condition,
                    "gold_gained_json": json.dumps(gained, ensure_ascii=False),
                    "gold_lost_json": json.dumps(lost, ensure_ascii=False),
                    "gold_found_by_both_json": json.dumps(shared, ensure_ascii=False),
                    "gold_found_by_neither_json": json.dumps(neither, ensure_ascii=False),
                    "predictions_added_json": json.dumps(
                        [variant_predictions[key] for key in added_keys], ensure_ascii=False
                    ),
                    "predictions_removed_json": json.dumps(
                        [base_predictions[key] for key in removed_keys], ensure_ascii=False
                    ),
                    "predictions_shared_json": json.dumps(
                        [variant_predictions[key] for key in shared_pred_keys], ensure_ascii=False
                    ),
                    "coverage_without": base["gold_coverage"],
                    "coverage_with": variant["gold_coverage"],
                    "coverage_delta": coverage_delta,
                    "coverage_outcome": coverage_outcome,
                    "informative_coverage_without": base.get("informative_gold_coverage"),
                    "informative_coverage_with": variant.get("informative_gold_coverage"),
                    "informative_coverage_delta": delta_optional("informative_gold_coverage"),
                    "ndcg_without": base["ndcg_all"],
                    "ndcg_with": variant["ndcg_all"],
                    "ndcg_delta": variant["ndcg_all"] - base["ndcg_all"],
                    "mrr_without": base["reciprocal_rank"],
                    "mrr_with": variant["reciprocal_rank"],
                    "mrr_delta": variant["reciprocal_rank"] - base["reciprocal_rank"],
                    "most_specific_rank_without": base_rank if base_rank is not None else "",
                    "most_specific_rank_with": variant_rank if variant_rank is not None else "",
                    "most_specific_change": most_specific_change,
                    "hit_anywhere_without": base["hit_anywhere"],
                    "hit_anywhere_with": variant["hit_anywhere"],
                    "hit_anywhere_delta": variant["hit_anywhere"] - base["hit_anywhere"],
                    "gold_types_json": base["gold_types_json"],
                }
            )
    return output


def build_pairwise_summary_rows(pairwise_rows):
    grouped = defaultdict(list)
    for row in pairwise_rows:
        grouped[(row["model"], row["comparison"], row["context_type"])].append(row)

    output = []
    for (model, comparison, context_type), rows in sorted(grouped.items()):
        change_counts = defaultdict(int)
        coverage_counts = defaultdict(int)
        for row in rows:
            change_counts[row["most_specific_change"]] += 1
            coverage_counts[row["coverage_outcome"]] += 1

        output.append(
            {
                "model": model,
                "comparison": comparison,
                "context_type": context_type,
                "entities": len(rows),
                "mean_coverage_delta": mean(row["coverage_delta"] for row in rows),
                "mean_informative_coverage_delta": mean_valid(
                    row["informative_coverage_delta"] for row in rows
                ),
                "mean_ndcg_delta": mean(row["ndcg_delta"] for row in rows),
                "mean_mrr_delta": mean(row["mrr_delta"] for row in rows),
                "mean_hit_anywhere_delta": mean(row["hit_anywhere_delta"] for row in rows),
                "coverage_improved_entities": coverage_counts["improved"],
                "coverage_unchanged_entities": coverage_counts["unchanged"],
                "coverage_harmed_entities": coverage_counts["harmed"],
                "most_specific_miss_to_hit": change_counts["miss_to_hit"],
                "most_specific_hit_to_miss": change_counts["hit_to_miss"],
                "both_hit_rank_improved": change_counts["both_hit_rank_improved"],
                "both_hit_rank_worsened": change_counts["both_hit_rank_worsened"],
                "both_hit_same_rank": change_counts["both_hit_same_rank"],
                "both_miss": change_counts["both_miss"],
                "total_gold_gained": sum(len(_json_list(row["gold_gained_json"])) for row in rows),
                "total_gold_lost": sum(len(_json_list(row["gold_lost_json"])) for row in rows),
            }
        )
    return output


def build_gold_type_change_rows(pairwise_rows):
    output = []
    for row in pairwise_rows:
        gold_types = _json_list(row["gold_types_json"])
        base_found = set(_json_list(row["gold_found_by_both_json"])) | set(
            _json_list(row["gold_lost_json"])
        )
        variant_found = set(_json_list(row["gold_found_by_both_json"])) | set(
            _json_list(row["gold_gained_json"])
        )
        max_depth = max((item["depth"] for item in gold_types), default=0)

        for item in gold_types:
            label = normalize_label(item["label"])
            base_hit = int(label in base_found)
            variant_hit = int(label in variant_found)
            if base_hit == 0 and variant_hit == 1:
                change = "gained"
            elif base_hit == 1 and variant_hit == 0:
                change = "lost"
            elif base_hit == 1:
                change = "both"
            else:
                change = "neither"

            output.append(
                {
                    "model": row["model"],
                    "comparison": row["comparison"],
                    "context_type": row["context_type"],
                    "entity_id": row["entity_id"],
                    "entity_label": row["entity_label"],
                    "gold_label": item["label"],
                    "gold_class_id": item["class_id"],
                    "gold_depth": item["depth"],
                    "depth_from_most_specific": max_depth - item["depth"],
                    "is_generic_analysis_type": int(
                        label in GENERIC_ANALYSIS_TYPES
                    ),
                    "without_hit": base_hit,
                    "with_hit": variant_hit,
                    "change": change,
                }
            )
    return output


def build_gold_depth_summary_rows(gold_change_rows):
    grouped = defaultdict(list)
    for row in gold_change_rows:
        key = (
            row["model"],
            row["comparison"],
            row["context_type"],
            row["depth_from_most_specific"],
        )
        grouped[key].append(row)

    output = []
    for key, rows in sorted(grouped.items()):
        model, comparison, context_type, distance = key
        output.append(
            {
                "model": model,
                "comparison": comparison,
                "context_type": context_type,
                "depth_from_most_specific": distance,
                "gold_labels": len(rows),
                "recall_without": mean(row["without_hit"] for row in rows),
                "recall_with": mean(row["with_hit"] for row in rows),
                "recall_delta": mean(row["with_hit"] - row["without_hit"] for row in rows),
                "gained_count": sum(row["change"] == "gained" for row in rows),
                "lost_count": sum(row["change"] == "lost" for row in rows),
            }
        )
    return output


def build_added_prediction_rows(pairwise_rows, prediction_rows):
    index = defaultdict(dict)
    for row in prediction_rows:
        key = (row["model"], row["entity_id"], row["condition"])
        index[key][normalize_label(row["prediction"])] = row

    output = []
    for pair in pairwise_rows:
        added = _json_list(pair["predictions_added_json"])
        condition_key = (
            pair["model"],
            pair["entity_id"],
            pair["with_condition"],
        )
        lookup = index.get(condition_key, {})
        for prediction in added:
            detail = lookup.get(normalize_label(prediction), {})
            output.append(
                {
                    "model": pair["model"],
                    "comparison": pair["comparison"],
                    "context_type": pair["context_type"],
                    "entity_id": pair["entity_id"],
                    "entity_label": pair["entity_label"],
                    "condition_added_by": pair["with_condition"],
                    "prediction": prediction,
                    "rank": detail.get("rank", ""),
                    "category": detail.get("category", ""),
                    "relation_to_gold": detail.get("relation_to_gold", ""),
                    "matched_class_ids_json": detail.get("matched_class_ids_json", "[]"),
                }
            )
    return output


def make_pairwise_plot(results_dir, pairwise_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = [
        row for row in pairwise_rows
        if row["comparison"] == "direct_oracle_taxonomy"
    ]
    if not rows:
        return

    rows = sorted(rows, key=lambda row: row["coverage_delta"])
    labels = [row["entity_label"] for row in rows]
    values = [row["coverage_delta"] for row in rows]
    y = list(range(len(rows)))

    plt.figure(figsize=(9, max(5, len(rows) * 0.28)))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Gold coverage change (oracle taxonomy - no taxonomy)")
    plt.title("Per-entity effect of oracle taxonomy context")
    plt.tight_layout()
    plt.savefig(results_dir / "taxonomy_coverage_delta_by_entity.png", dpi=200)
    plt.close()


def make_plots(results_dir, hit_rows, distribution_rows, category_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; CSV outputs were created but plots were skipped.")
        print("Install it with: python3 -m pip install matplotlib")
        return

    # Cumulative hit rate for the most-specific gold type.
    by_condition = defaultdict(list)
    for row in hit_rows:
        by_condition[row["condition"]].append(row)

    plt.figure(figsize=(8, 5))
    for condition, rows in sorted(by_condition.items()):
        rows = sorted(rows, key=lambda item: item["rank_cutoff"])
        plt.plot(
            [row["rank_cutoff"] for row in rows],
            [row["hit_rate"] for row in rows],
            marker="o",
            label=condition,
        )
    plt.xlabel("Rank cutoff")
    plt.ylabel("Hit rate")
    plt.title("Most-specific gold type: cumulative hit rate by rank")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "hit_rate_by_rank.png", dpi=200)
    plt.close()

    # Distribution of per-entity gold coverage.
    bucket_order = [item[2] for item in COVERAGE_BUCKETS]
    conditions = sorted({row["condition"] for row in distribution_rows})
    x = list(range(len(bucket_order)))
    width = 0.8 / max(len(conditions), 1)

    plt.figure(figsize=(9, 5))
    for index, condition in enumerate(conditions):
        lookup = {
            row["coverage_bucket"]: row["entity_share"]
            for row in distribution_rows
            if row["condition"] == condition
        }
        positions = [value - 0.4 + width / 2 + index * width for value in x]
        plt.bar(
            positions,
            [lookup.get(bucket, 0.0) for bucket in bucket_order],
            width=width,
            label=condition,
        )
    plt.xticks(x, bucket_order)
    plt.xlabel("Gold types recovered per entity")
    plt.ylabel("Share of entities")
    plt.title("Distribution of gold-type coverage")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "gold_coverage_distribution.png", dpi=200)
    plt.close()

    if any(
        row["category"] not in {"gold", "non_gold_unchecked"}
        for row in category_rows
    ):
        categories = [
            "gold",
            "broader_related_extra",
            "narrower_related_extra",
            "sibling_related_extra",
            "other_wikc_extra",
            "out_of_taxonomy",
        ]
        conditions = sorted({row["condition"] for row in category_rows})
        lookup = defaultdict(dict)
        for row in category_rows:
            lookup[row["condition"]][row["category"]] = row["prediction_share"]

        plt.figure(figsize=(9, 5))
        bottoms = [0.0] * len(conditions)
        for category in categories:
            values = [lookup[condition].get(category, 0.0) for condition in conditions]
            plt.bar(conditions, values, bottom=bottoms, label=category)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        plt.ylabel("Share of predictions")
        plt.title("Prediction categories: gold, related extras, and hallucinations")
        plt.ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(results_dir / "prediction_category_distribution.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score ranked entity-type predictions, measure full gold coverage, "
            "analyze additional predictions against WiKC, and plot hit rates."
        )
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--aliases", type=Path, default=None)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("data/wikc.tsv"),
        help="WiKC child->parent edge file used to analyze extra predictions.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/wiki_taxonomy_labels.tsv"),
        help="WiKC label file used to detect whether extra labels exist in the taxonomy.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    args = parser.parse_args()

    aliases = load_aliases(args.aliases)
    taxonomy_resources = None

    if args.taxonomy.exists() and args.labels.exists():
        print("Loading WiKC graph and label index for extra-type analysis...")
        graph = build_graph(args.taxonomy)
        label_to_ids, id_to_label = load_label_index(args.labels)
        taxonomy_resources = (graph, label_to_ids, id_to_label)
        print(f"  graph classes: {graph.number_of_nodes()}")
        print(f"  graph edges:   {graph.number_of_edges()}")
        print(f"  labels:        {len(id_to_label)}")
    else:
        print("Warning: WiKC taxonomy/label files were not found.")
        print("Gold coverage and ranking will be scored, but hallucination classification will be skipped.")

    result_files = sorted(args.results.glob("example_*/*_result.json"))
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found under {args.results}")

    detailed_rows = []
    prediction_rows = []

    for result_file in result_files:
        detailed, predictions = score_result_file(
            result_file,
            aliases,
            taxonomy_resources,
        )
        if detailed is not None:
            detailed_rows.append(detailed)
            prediction_rows.extend(predictions)

    if not detailed_rows:
        raise RuntimeError("No completed results were available for scoring.")

    summary_rows = aggregate_conditions(
        detailed_rows,
        prediction_rows,
        taxonomy_resources is not None,
    )
    distribution_rows = build_distribution_rows(detailed_rows)
    hit_rows = build_hit_by_rank_rows(detailed_rows)
    category_rows = build_prediction_category_rows(prediction_rows)
    comparison_rows = build_comparison_rows(summary_rows)
    pairwise_rows = build_pairwise_taxonomy_rows(detailed_rows)
    pairwise_summary_rows = build_pairwise_summary_rows(pairwise_rows)
    gold_change_rows = build_gold_type_change_rows(pairwise_rows)
    gold_depth_rows = build_gold_depth_summary_rows(gold_change_rows)
    added_prediction_rows = build_added_prediction_rows(pairwise_rows, prediction_rows)

    detailed_fields = list(detailed_rows[0].keys())
    prediction_fields = list(prediction_rows[0].keys()) if prediction_rows else []
    summary_fields = list(summary_rows[0].keys())

    write_csv(
        args.results / "weighted_detailed_scores.csv",
        detailed_rows,
        detailed_fields,
    )
    write_csv(
        args.results / "prediction_level_analysis.csv",
        prediction_rows,
        prediction_fields,
    )
    write_csv(
        args.results / "weighted_condition_scores.csv",
        summary_rows,
        summary_fields,
    )
    write_csv(
        args.results / "gold_coverage_distribution.csv",
        distribution_rows,
        list(distribution_rows[0].keys()),
    )
    write_csv(
        args.results / "hit_rate_by_rank.csv",
        hit_rows,
        list(hit_rows[0].keys()),
    )
    write_csv(
        args.results / "prediction_category_distribution.csv",
        category_rows,
        list(category_rows[0].keys()),
    )

    if comparison_rows:
        write_csv(
            args.results / "comparison_to_standard.csv",
            comparison_rows,
            list(comparison_rows[0].keys()),
        )

    if pairwise_rows:
        write_csv(
            args.results / "taxonomy_pairwise_analysis.csv",
            pairwise_rows,
            list(pairwise_rows[0].keys()),
        )
        write_csv(
            args.results / "taxonomy_effect_summary.csv",
            pairwise_summary_rows,
            list(pairwise_summary_rows[0].keys()),
        )
    if gold_change_rows:
        write_csv(
            args.results / "taxonomy_gold_type_changes.csv",
            gold_change_rows,
            list(gold_change_rows[0].keys()),
        )
        write_csv(
            args.results / "taxonomy_gold_depth_summary.csv",
            gold_depth_rows,
            list(gold_depth_rows[0].keys()),
        )
    if added_prediction_rows:
        write_csv(
            args.results / "taxonomy_added_predictions.csv",
            added_prediction_rows,
            list(added_prediction_rows[0].keys()),
        )

    if not args.no_plots:
        make_plots(
            args.results,
            hit_rows,
            distribution_rows,
            category_rows,
        )
        make_pairwise_plot(args.results, pairwise_rows)

    print("\nScoring complete.\n")
    for row in summary_rows:
        print(f"{row['model']} | {row['condition']}")
        print(f"  Mean gold coverage: {row['mean_gold_coverage']:.3f}")
        print(f"  nDCG (all):        {row['mean_ndcg_all']:.3f}")
        print(f"  MRR:               {row['mrr']:.3f}")
        print(f"  Hit anywhere:      {row['hit_rate_anywhere']:.3f}")
        informative = row.get("mean_informative_gold_coverage")
        if informative not in (None, ""):
            print(f"  Informative coverage: {informative:.3f}")
        if row["out_of_taxonomy_hallucination_rate"] != "":
            print(
                "  Hallucination rate: "
                f"{row['out_of_taxonomy_hallucination_rate']:.3f}"
            )

    print("\nOutputs written to:", args.results)
    print("  weighted_detailed_scores.csv")
    print("  weighted_condition_scores.csv")
    print("  prediction_level_analysis.csv")
    print("  gold_coverage_distribution.csv")
    print("  hit_rate_by_rank.csv")
    print("  comparison_to_standard.csv")
    if pairwise_rows:
        print("  taxonomy_pairwise_analysis.csv")
        print("  taxonomy_effect_summary.csv")
        print("  taxonomy_gold_type_changes.csv")
        print("  taxonomy_gold_depth_summary.csv")
        print("  taxonomy_added_predictions.csv")
    if not args.no_plots:
        print("  hit_rate_by_rank.png")
        print("  gold_coverage_distribution.png")
        if taxonomy_resources is not None:
            print("  prediction_category_distribution.png")
        if pairwise_rows:
            print("  taxonomy_coverage_delta_by_entity.png")


if __name__ == "__main__":
    main()
