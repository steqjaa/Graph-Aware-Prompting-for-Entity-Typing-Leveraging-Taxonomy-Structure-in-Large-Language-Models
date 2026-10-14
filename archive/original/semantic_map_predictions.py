from pathlib import Path
import argparse
import csv
import json
import re
from collections import defaultdict

import networkx as nx
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit(
        "  python3 -m pip install sentence-transformers"
    ) from error


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MIN_MARGIN = 0.05

LABEL_PATTERN = re.compile(
    r'^(wd:Q\d+)\s+rdfs:label\s+"((?:\\.|[^"])*)"\s+\.$'
)


def normalize_label(text):
    text = text.casefold().strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def build_graph(path):
    graph = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                graph.add_edge(row[0], row[1])
    return graph


def load_graph_labels(path, graph):
    wanted = set(graph.nodes())
    id_to_label = {}
    label_to_ids = defaultdict(set)

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            match = LABEL_PATTERN.match(line.strip())
            if not match:
                continue
            class_id = match.group(1)
            if class_id not in wanted:
                continue
            label = match.group(2).replace(r'\"', '"')
            id_to_label.setdefault(class_id, label)
            label_to_ids[normalize_label(label)].add(class_id)

    return id_to_label, dict(label_to_ids)


def collect_predictions(results_dir):
    records = []
    unique = {}

    for path in sorted(results_dir.glob("example_*/*_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        for rank, prediction in enumerate(result.get("predictions", []), start=1):
            normalized = normalize_label(prediction)
            if not normalized:
                continue
            records.append(
                {
                    "example_number": result.get("example_number"),
                    "entity_id": result.get("entity_id"),
                    "entity_label": result.get("entity_label"),
                    "condition": result.get("condition"),
                    "rank": rank,
                    "prediction": prediction,
                    "normalized_prediction": normalized,
                }
            )
            unique.setdefault(normalized, prediction)

    return records, unique


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Map free-form LLM type labels to WiKC classes using exact labels "
            "first and sentence-embedding similarity as a fallback."
        )
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("data/wikc.tsv"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/wiki_taxonomy_labels.tsv"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    graph = build_graph(args.taxonomy)
    id_to_label, label_to_ids = load_graph_labels(args.labels, graph)

    class_ids = sorted(id_to_label)
    class_labels = [id_to_label[class_id] for class_id in class_ids]

    print(f"WiKC graph classes: {graph.number_of_nodes()}")
    print(f"WiKC class labels:  {len(class_labels)}")

    prediction_records, unique_predictions = collect_predictions(args.results)
    print(f"Unique predicted labels: {len(unique_predictions)}")

    model = SentenceTransformer(args.embedding_model)
    print("Encoding WiKC class labels...")
    class_embeddings = model.encode(
        class_labels,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    mapping = {}
    unresolved = []

    semantic_queries = [
        (normalized, original)
        for normalized, original in unique_predictions.items()
        if normalized not in label_to_ids
    ]

    semantic_embeddings = None
    if semantic_queries:
        semantic_embeddings = model.encode(
            [original for _, original in semantic_queries],
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    semantic_index = {
        normalized: semantic_embeddings[index]
        for index, (normalized, _) in enumerate(semantic_queries)
    } if semantic_queries else {}

    for normalized, original in unique_predictions.items():
        exact_ids = sorted(label_to_ids.get(normalized, set()))
        if exact_ids:
            class_id = exact_ids[0]
            mapping[normalized] = {
                "prediction": original,
                "method": "exact",
                "accepted": True,
                "mapped_label": id_to_label[class_id],
                "mapped_class_id": class_id,
                "similarity": 1.0,
                "second_similarity": None,
                "margin": None,
                "top_candidates": [
                    {
                        "class_id": item,
                        "label": id_to_label[item],
                        "similarity": 1.0,
                    }
                    for item in exact_ids[: args.top_k]
                ],
            }
            continue

        query_embedding = semantic_index[normalized]
        scores = class_embeddings @ query_embedding
        top_indices = np.argsort(-scores)[: args.top_k]
        candidates = [
            {
                "class_id": class_ids[index],
                "label": class_labels[index],
                "similarity": float(scores[index]),
            }
            for index in top_indices
        ]

        top = candidates[0]
        second = candidates[1]["similarity"] if len(candidates) > 1 else None
        margin = top["similarity"] - second if second is not None else None
        accepted = (
            top["similarity"] >= args.threshold
            and (margin is None or margin >= args.min_margin)
        )

        mapping[normalized] = {
            "prediction": original,
            "method": "embedding" if accepted else "unmatched",
            "accepted": accepted,
            "mapped_label": top["label"] if accepted else "",
            "mapped_class_id": top["class_id"] if accepted else "",
            "similarity": top["similarity"],
            "second_similarity": second,
            "margin": margin,
            "top_candidates": candidates,
        }
        if not accepted:
            unresolved.append(original)

    mapping_rows = []
    for normalized, item in sorted(mapping.items()):
        mapping_rows.append(
            {
                "prediction": item["prediction"],
                "normalized_prediction": normalized,
                "mapping_method": item["method"],
                "accepted": item["accepted"],
                "mapped_label": item["mapped_label"],
                "mapped_class_id": item["mapped_class_id"],
                "similarity": item["similarity"],
                "second_similarity": item["second_similarity"],
                "margin": item["margin"],
                "top_candidates_json": json.dumps(
                    item["top_candidates"], ensure_ascii=False
                ),
            }
        )

    occurrence_rows = []
    for record in prediction_records:
        item = mapping[record["normalized_prediction"]]
        occurrence_rows.append(
            {
                **record,
                "mapping_method": item["method"],
                "accepted": item["accepted"],
                "mapped_label": item["mapped_label"],
                "mapped_class_id": item["mapped_class_id"],
                "similarity": item["similarity"],
                "margin": item["margin"],
            }
        )

    alias_map = {
        item["prediction"]: item["mapped_label"]
        for item in mapping.values()
        if item["method"] == "embedding" and item["accepted"]
    }

    mapping_csv = args.results / "semantic_mapping_candidates.csv"
    occurrences_csv = args.results / "semantic_mapping_occurrences.csv"
    aliases_json = args.results / "semantic_aliases_high_confidence.json"
    report_json = args.results / "semantic_mapping_report.json"

    with open(mapping_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mapping_rows)

    if occurrence_rows:
        with open(occurrences_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(occurrence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(occurrence_rows)

    aliases_json.write_text(
        json.dumps(alias_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_json.write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "threshold": args.threshold,
                "min_margin": args.min_margin,
                "unique_predictions": len(unique_predictions),
                "exact_mappings": sum(
                    item["method"] == "exact" for item in mapping.values()
                ),
                "accepted_embedding_mappings": sum(
                    item["method"] == "embedding" for item in mapping.values()
                ),
                "unmatched_predictions": len(unresolved),
                "warning": (
                    "Embedding threshold and margin are exploratory defaults. "
                    "Calibrate them on development data before final test scoring."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSemantic mapping complete.")
    print("Candidate mappings:", mapping_csv)
    print("High-confidence aliases:", aliases_json)
    print("Report:", report_json)
    print(
        "Important: calibrate --threshold and --min-margin on development data "
        "before using semantic aliases for final reported metrics."
    )


if __name__ == "__main__":
    main()
