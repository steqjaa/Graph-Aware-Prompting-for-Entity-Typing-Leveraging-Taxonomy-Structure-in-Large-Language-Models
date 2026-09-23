from pathlib import Path
import argparse
import csv
import json
import re

import networkx as nx
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as error:
    raise SystemExit(
        "sentence-transformers is required. Install it with:\n"
        "  python3 -m pip install sentence-transformers"
    ) from error

from generate_ranked_prompts import (
    COMMON_TASK,
    GRAPH_NOTE,
    COT_GRAPH_INSTRUCTIONS,
    CONTRASTIVE_GRAPH_INSTRUCTIONS,
    compact_json_types,
    compact_json_reasoned,
    build_graph,
    get_graph_context,
    load_selected_labels,
    ranked_gold_types,
    read_entities,
    readable_list,
    select_demonstration_entity,
)


DATA = Path("data")
DEFAULT_PROMPT_FOLDER = Path("generated_ranked_prompts_v3")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
MAX_SIBLINGS = 5
MAX_CHILDREN = 5

LABEL_PATTERN = re.compile(
    r'^(wd:Q\d+)\s+rdfs:label\s+"((?:\\.|[^"])*)"\s+\.$'
)


def normalize_label(text):
    text = text.casefold().strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def load_graph_labels(label_file, graph):
    wanted = set(graph.nodes())
    id_to_label = {}

    with open(label_file, "r", encoding="utf-8") as file:
        for line in file:
            match = LABEL_PATTERN.match(line.strip())
            if not match:
                continue
            class_id = match.group(1)
            if class_id not in wanted:
                continue
            label = match.group(2).replace(r'\"', '"')
            if class_id not in id_to_label:
                id_to_label[class_id] = label
            if len(id_to_label) == len(wanted):
                break

    return id_to_label


def embedding_cache_path(prompt_folder, model_name):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return prompt_folder / f"retrieval_index__{safe}.npz"


def build_or_load_index(model, graph, id_to_label, prompt_folder, model_name):
    cache_path = embedding_cache_path(prompt_folder, model_name)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        class_ids = data["class_ids"].tolist()
        labels = data["labels"].tolist()
        embeddings = data["embeddings"]
        print(f"Loaded cached retrieval index: {cache_path}")
        return class_ids, labels, embeddings

    class_ids = sorted(
        class_id
        for class_id in graph.nodes()
        if class_id in id_to_label
    )
    labels = [id_to_label[class_id] for class_id in class_ids]

    print(f"Encoding {len(labels)} WiKC class labels...")
    embeddings = model.encode(
        labels,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.savez_compressed(
        cache_path,
        class_ids=np.asarray(class_ids, dtype=object),
        labels=np.asarray(labels, dtype=object),
        embeddings=embeddings,
    )
    print(f"Saved retrieval index: {cache_path}")
    return class_ids, labels, embeddings


def retrieve_candidates(model, query, class_ids, labels, embeddings, top_k):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]

    return [
        {
            "class_id": class_ids[index],
            "label": labels[index],
            "similarity": float(scores[index]),
        }
        for index in top_indices
    ]


def candidate_neighborhood(graph, class_id, labels):
    context = get_graph_context(graph, class_id)
    return {
        "retrieved_center_class_id": class_id,
        "retrieved_center_label": labels.get(class_id, class_id),
        "parents": context["parents"],
        "siblings": context["siblings"][:MAX_SIBLINGS],
        "children": context["children"][:MAX_CHILDREN],
        "parents_readable": readable_list(context["parents"], labels),
        "siblings_readable": readable_list(context["siblings"][:MAX_SIBLINGS], labels),
        "children_readable": readable_list(context["children"][:MAX_CHILDREN], labels),
    }


def format_retrieved_context(neighborhoods):
    blocks = []
    for index, item in enumerate(neighborhoods, start=1):
        # Deliberately omit the retrieved center label from the model prompt.
        # The retrieval system chose the center from the description; the LLM
        # receives only its surrounding graph structure, matching the oracle
        # experiment as closely as possible.
        blocks.append(
            "\n".join(
                [
                    f"Retrieved taxonomy neighborhood {index}:",
                    f"- Direct parent classes: {item['parents_readable']}",
                    f"- Sibling classes: {item['siblings_readable']}",
                    f"- Direct child classes: {item['children_readable']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def retrieved_note():
    return """
The taxonomy neighborhoods below were retrieved automatically from the
entity description using semantic similarity. They may be useful or noisy.
The center class of each neighborhood is omitted. Use the structural clues
only when they agree with the entity description. Do not assume that every
retrieved neighborhood is relevant, and do not copy a parent, sibling, or
child merely because it appears in the context.
""".strip()


def create_retrieved_prompts(
    example_entity,
    example_ranking,
    example_context_text,
    test_metadata,
    test_context_text,
):
    no_graph_wrong_ranking = [
        "producer",
        "creative producer",
        "co-producer",
        "line producer",
        "radio producer",
        "DJ producer",
        "professional",
        "worker",
        "person",
        "entity",
    ]

    example_explanation = (
        f'The description "{example_entity["description"]}" supports '
        f'{example_ranking[0]} as the most specific type. The automatically '
        'retrieved neighborhoods provide structural evidence that can refine '
        'the decision, but only where it agrees with the description. Sibling '
        'and child labels are alternatives or narrower possibilities rather '
        'than automatic answers. The final ranking proceeds from the most '
        'specific supported type to broader valid types.'
    )

    wrong_explanation = (
        "This ranking is wrong because it treats retrieved neighboring classes "
        "as guaranteed answers and promotes labels that are not independently "
        "supported by the entity description."
    )

    direct_prompt = f"""
{COMMON_TASK}

{retrieved_note()}

{GRAPH_NOTE}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Automatically retrieved taxonomy context:
{example_context_text}

Answer:
{compact_json_types(example_ranking)}

Now classify:

Entity: {test_metadata["entity_label"]}
Description: {test_metadata["description"]}

Automatically retrieved taxonomy context:
{test_context_text}

Answer:
""".strip()

    cot_prompt = f"""
{COMMON_TASK}

{retrieved_note()}

{GRAPH_NOTE}

{COT_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "retrieval- and taxonomy-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Automatically retrieved taxonomy context:
{example_context_text}

Correct response:
{compact_json_reasoned(example_explanation, example_ranking)}

Now classify:

Entity: {test_metadata["entity_label"]}
Description: {test_metadata["description"]}

Automatically retrieved taxonomy context:
{test_context_text}

Answer:
""".strip()

    contrastive_prompt = f"""
{COMMON_TASK}

{retrieved_note()}

{GRAPH_NOTE}

{COT_GRAPH_INSTRUCTIONS}

{CONTRASTIVE_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "contrastive retrieval- and taxonomy-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Automatically retrieved taxonomy context:
{example_context_text}

Correct response:
{compact_json_reasoned(example_explanation, example_ranking)}

Wrong ranking:
{compact_json_types(no_graph_wrong_ranking)}

Why the wrong ranking is wrong:
{wrong_explanation}

Now classify:

Entity: {test_metadata["entity_label"]}
Description: {test_metadata["description"]}

Automatically retrieved taxonomy context:
{test_context_text}

Answer:
""".strip()

    return {
        "retrieved_graph_standard": direct_prompt,
        "cot_retrieved_graph": cot_prompt,
        "contrastive_cot_retrieved_graph": contrastive_prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append automatically retrieved graph-context prompt conditions to "
            "an existing generated prompt folder. Retrieval uses only entity "
            "label/description, never the test gold type."
        )
    )
    parser.add_argument(
        "--prompt-folder",
        type=Path,
        default=DEFAULT_PROMPT_FOLDER,
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DATA / "wikc.tsv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA / "wiki_taxonomy_labels.tsv",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=DATA / "wikc_eval_1k.txt",
    )
    args = parser.parse_args()

    prompt_folder = args.prompt_folder
    manifest_path = prompt_folder / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    graph = build_graph(args.taxonomy)
    graph_labels = load_graph_labels(args.labels, graph)
    print(f"Graph classes: {graph.number_of_nodes()}")
    print(f"Graph labels available: {len(graph_labels)}")

    model = SentenceTransformer(args.embedding_model)
    class_ids, class_labels, embeddings = build_or_load_index(
        model,
        graph,
        graph_labels,
        prompt_folder,
        args.embedding_model,
    )

    # Demonstration answer may use gold labels, as in the existing few-shot
    # prompts, but its retrieved context is produced from its text alone.
    entities = read_entities(args.evaluation)
    example_entity, _ = select_demonstration_entity(entities, graph)
    demo_required = {
        item["class_id"] for item in example_entity["taxonomy_classes"]
    }
    demo_labels = load_selected_labels(args.labels, demo_required)
    combined_labels = dict(graph_labels)
    combined_labels.update(demo_labels)
    example_ranking = [
        item["label"]
        for item in ranked_gold_types(example_entity, combined_labels)
    ]

    demo_query = f'{example_entity["entity_label"]}. {example_entity["description"]}'
    demo_candidates = retrieve_candidates(
        model,
        demo_query,
        class_ids,
        class_labels,
        embeddings,
        args.top_k,
    )
    demo_neighborhoods = []
    for candidate in demo_candidates:
        neighborhood = candidate_neighborhood(
            graph, candidate["class_id"], combined_labels
        )
        neighborhood["similarity"] = candidate["similarity"]
        demo_neighborhoods.append(neighborhood)
    demo_context_text = format_retrieved_context(demo_neighborhoods)

    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        manifest_rows = list(csv.DictReader(file))

    retrieved_conditions = {
        "retrieved_graph_standard",
        "cot_retrieved_graph",
        "contrastive_cot_retrieved_graph",
    }
    manifest_rows = [
        row for row in manifest_rows
        if row["condition"] not in retrieved_conditions
    ]

    example_folders = sorted(prompt_folder.glob("example_*"))
    added_rows = []

    for example_folder in example_folders:
        metadata_path = example_folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        query = f'{metadata["entity_label"]}. {metadata["description"]}'
        candidates = retrieve_candidates(
            model,
            query,
            class_ids,
            class_labels,
            embeddings,
            args.top_k,
        )

        neighborhoods = []
        for candidate in candidates:
            neighborhood = candidate_neighborhood(
                graph, candidate["class_id"], combined_labels
            )
            neighborhood["similarity"] = candidate["similarity"]
            neighborhoods.append(neighborhood)

        context_text = format_retrieved_context(neighborhoods)
        prompts = create_retrieved_prompts(
            example_entity,
            example_ranking,
            demo_context_text,
            metadata,
            context_text,
        )

        metadata["retrieval"] = {
            "method": "sentence_embedding_label_retrieval",
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "query": query,
            "candidates": candidates,
            "neighborhoods": neighborhoods,
            "gold_used_for_test_retrieval": False,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        example_number = int(metadata["example_number"])
        for condition, prompt in prompts.items():
            prompt_path = example_folder / f"{condition}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            added_rows.append(
                {
                    "example_number": example_number,
                    "entity_id": metadata["entity_id"],
                    "entity_label": metadata["entity_label"],
                    "description": metadata["description"],
                    "condition": condition,
                    "prompt_path": str(prompt_path),
                    "metadata_path": str(metadata_path),
                }
            )

        print(
            f"{example_number:02d}. {metadata['entity_label']} | "
            + ", ".join(
                f"{item['label']}={item['similarity']:.3f}"
                for item in candidates
            )
        )

    manifest_rows.extend(added_rows)
    manifest_rows.sort(
        key=lambda row: (int(row["example_number"]), row["condition"])
    )

    fieldnames = [
        "example_number",
        "entity_id",
        "entity_label",
        "description",
        "condition",
        "prompt_path",
        "metadata_path",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    retrieval_report = prompt_folder / "retrieval_report.json"
    retrieval_report.write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "top_k": args.top_k,
                "graph_classes": graph.number_of_nodes(),
                "graph_labels": len(graph_labels),
                "test_gold_used_for_retrieval": False,
                "retrieved_conditions_added": sorted(retrieved_conditions),
                "manifest_rows": len(manifest_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRetrieved-context prompts added.")
    print("Manifest:", manifest_path)
    print("Total manifest rows:", len(manifest_rows))
    print("Report:", retrieval_report)


if __name__ == "__main__":
    main()
