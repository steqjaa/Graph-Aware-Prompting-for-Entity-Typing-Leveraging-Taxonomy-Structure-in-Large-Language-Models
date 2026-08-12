from pathlib import Path
import argparse
import csv
import json
import math
import re
from collections import defaultdict


TOP_K = 10


# =========================================================
# Matching and metric helpers
# =========================================================

def normalize_label(text):
    text = text.casefold().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def load_aliases(alias_file):
    """
    Optional JSON format:

    {
      "coastline": "coast",
      "tanka poet": "waka poet"
    }

    Keep this file fixed before the final evaluation run.
    """
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


def relevance_lookup(gold_types):
    """
    Assign hierarchical relevance from WiKC depth:

        relevance = depth / maximum entity depth

    The most specific listed class or classes receive 1.0.
    Broader valid ancestors receive partial credit.
    Unmatched predictions receive 0.
    """
    maximum_depth = max(
        item["depth"]
        for item in gold_types
    )

    lookup = {}

    for item in gold_types:
        label = normalize_label(item["label"])

        if maximum_depth > 0:
            relevance = item["depth"] / maximum_depth
        else:
            relevance = 1.0

        lookup[label] = max(
            lookup.get(label, 0.0),
            relevance,
        )

    most_specific_labels = {
        normalize_label(item["label"])
        for item in gold_types
        if item["depth"] == maximum_depth
    }

    return lookup, most_specific_labels


def dcg_at_k(relevances, k):
    return sum(
        (2.0 ** relevance - 1.0)
        / math.log2(rank + 1)
        for rank, relevance in enumerate(
            relevances[:k],
            start=1,
        )
    )


def ndcg_at_k(predictions, gold_types, aliases, k=TOP_K):
    gold_relevance, _ = relevance_lookup(gold_types)

    observed_relevances = []
    matched_gold_labels = set()

    for prediction in predictions[:k]:
        canonical = canonical_prediction(
            prediction,
            aliases,
        )

        relevance = gold_relevance.get(canonical, 0.0)

        # Repeating the same valid class does not earn extra gain.
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


def reciprocal_rank_at_k(
    predictions,
    gold_types,
    aliases,
    k=TOP_K,
):
    """
    MRR uses the first occurrence of any maximum-depth WiKC type.
    """
    _, most_specific_labels = relevance_lookup(gold_types)

    for rank, prediction in enumerate(
        predictions[:k],
        start=1,
    ):
        canonical = canonical_prediction(
            prediction,
            aliases,
        )

        if canonical in most_specific_labels:
            return 1.0 / rank

    return 0.0


def hit_at_k(
    predictions,
    gold_types,
    aliases,
    k=TOP_K,
):
    _, most_specific_labels = relevance_lookup(gold_types)

    return int(
        any(
            canonical_prediction(prediction, aliases)
            in most_specific_labels
            for prediction in predictions[:k]
        )
    )


# =========================================================
# Scoring pipeline
# =========================================================

def score_result_file(result_file, aliases):
    result = json.loads(
        result_file.read_text(encoding="utf-8")
    )

    if result.get("status") != "completed":
        return None

    metadata_path = Path(result["metadata_path"])
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    predictions = result.get("predictions", [])
    gold_types = metadata["gold_types"]

    reciprocal_rank = reciprocal_rank_at_k(
        predictions,
        gold_types,
        aliases,
    )

    ndcg = ndcg_at_k(
        predictions,
        gold_types,
        aliases,
    )

    hit = hit_at_k(
        predictions,
        gold_types,
        aliases,
    )

    return {
        "example_number": result["example_number"],
        "entity_id": result["entity_id"],
        "entity_label": result["entity_label"],
        "condition": result["condition"],
        "model": result["model"],
        "reference_type": metadata["reference_type"],
        "predictions_json": json.dumps(
            predictions,
            ensure_ascii=False,
        ),
        "reciprocal_rank_at_10": reciprocal_rank,
        "ndcg_at_10": ndcg,
        "hit_at_10": hit,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score ranked entity-type predictions with "
            "MRR@10 and nDCG@10."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help=(
            "Model result folder, for example "
            "generated_ranked_results/llama-3.1-8b-instant"
        ),
    )

    parser.add_argument(
        "--aliases",
        type=Path,
        default=None,
        help=(
            "Optional JSON alias mapping fixed before final scoring."
        ),
    )

    args = parser.parse_args()
    aliases = load_aliases(args.aliases)

    result_files = sorted(
        args.results.glob(
            "example_*/*_result.json"
        )
    )

    if not result_files:
        raise FileNotFoundError(
            f"No result JSON files found under {args.results}"
        )

    detailed_rows = []

    for result_file in result_files:
        scored = score_result_file(
            result_file,
            aliases,
        )

        if scored is not None:
            detailed_rows.append(scored)

    if not detailed_rows:
        raise RuntimeError(
            "No completed results were available for scoring."
        )

    aggregates = defaultdict(
        lambda: {
            "count": 0,
            "sum_rr": 0.0,
            "sum_ndcg": 0.0,
            "sum_hit": 0,
        }
    )

    for row in detailed_rows:
        key = (row["model"], row["condition"])
        aggregate = aggregates[key]

        aggregate["count"] += 1
        aggregate["sum_rr"] += row[
            "reciprocal_rank_at_10"
        ]
        aggregate["sum_ndcg"] += row["ndcg_at_10"]
        aggregate["sum_hit"] += row["hit_at_10"]

    summary_rows = []

    for (model, condition), aggregate in sorted(
        aggregates.items()
    ):
        count = aggregate["count"]

        summary_rows.append(
            {
                "model": model,
                "condition": condition,
                "entities_scored": count,
                "mrr_at_10": (
                    aggregate["sum_rr"] / count
                ),
                "mean_ndcg_at_10": (
                    aggregate["sum_ndcg"] / count
                ),
                "hit_rate_at_10": (
                    aggregate["sum_hit"] / count
                ),
            }
        )

    detailed_path = (
        args.results / "weighted_detailed_scores.csv"
    )
    summary_path = (
        args.results / "weighted_condition_scores.csv"
    )

    with open(
        detailed_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "example_number",
            "entity_id",
            "entity_label",
            "condition",
            "model",
            "reference_type",
            "predictions_json",
            "reciprocal_rank_at_10",
            "ndcg_at_10",
            "hit_at_10",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    with open(
        summary_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "model",
            "condition",
            "entities_scored",
            "mrr_at_10",
            "mean_ndcg_at_10",
            "hit_rate_at_10",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Weighted scoring complete.\n")

    for row in summary_rows:
        print(
            f"{row['model']} | {row['condition']}\n"
            f"  MRR@10:  {row['mrr_at_10']:.4f}\n"
            f"  nDCG@10: {row['mean_ndcg_at_10']:.4f}\n"
            f"  Hit@10:  {row['hit_rate_at_10']:.4f}"
        )

    print("\nDetailed scores:", detailed_path)
    print("Condition summary:", summary_path)


if __name__ == "__main__":
    main()