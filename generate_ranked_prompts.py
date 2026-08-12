from pathlib import Path
import csv
import json
import re

import networkx as nx


# =========================================================
# Configuration
# =========================================================

DATA = Path("data")
OUTPUT = Path("generated_ranked_prompts")

EVALUATION_FILE = DATA / "wikc_eval_1k.txt"
TAXONOMY_FILE = DATA / "wikc.tsv"
LABELS_FILE = DATA / "wiki_taxonomy_labels.tsv"

NUMBER_OF_TEST_ENTITIES = 20

# Keep 0 while debugging. Set this to 20 for the next fresh evaluation
# set, because the first 20 entities were already used to refine prompts.
SKIP_USABLE_TEST_ENTITIES = 20

MAX_SIBLINGS = 5
MAX_CHILDREN = 5
TOP_K = 10


# =========================================================
# 1. Read WiKC evaluation entities
# =========================================================

def read_entities(file_path):
    entities = {}

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(
            file,
            delimiter="\t",
            quotechar="'",
        )

        for line_number, row in enumerate(reader, start=1):
            if len(row) != 5:
                print(
                    f"Warning: skipped malformed evaluation row "
                    f"{line_number}: {row}"
                )
                continue

            entity_id, entity_label, description, class_id, depth = row

            try:
                depth = int(depth)
            except ValueError:
                print(
                    f"Warning: skipped row {line_number}; "
                    f"invalid depth: {row}"
                )
                continue

            if entity_id not in entities:
                entities[entity_id] = {
                    "entity_id": entity_id,
                    "entity_label": entity_label,
                    "description": description,
                    "taxonomy_classes": [],
                }

            entities[entity_id]["taxonomy_classes"].append(
                {
                    "class_id": class_id,
                    "depth": depth,
                }
            )

    result = list(entities.values())

    if len(result) < NUMBER_OF_TEST_ENTITIES + 1:
        raise RuntimeError(
            "The evaluation file does not contain enough entities."
        )

    return result


# =========================================================
# 2. Build WiKC graph
# =========================================================

def build_graph(file_path):
    graph = nx.DiGraph()

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                print(
                    f"Warning: skipped malformed taxonomy row "
                    f"{line_number}: {row}"
                )
                continue

            child, parent = row
            graph.add_edge(child, parent)

    return graph


# =========================================================
# 3. Select usable entities
# =========================================================

def find_most_specific_types(graph, entity):
    """
    Return the listed entity classes for which no more-specific listed
    class reaches them through child -> parent edges.
    """
    entity_type_ids = {
        item["class_id"]
        for item in entity["taxonomy_classes"]
    }

    most_specific_types = []

    for class_id in entity_type_ids:
        if class_id not in graph:
            more_specific_entity_types = set()
        else:
            more_specific_entity_types = (
                nx.ancestors(graph, class_id) & entity_type_ids
            )

        if not more_specific_entity_types:
            most_specific_types.append(class_id)

    return sorted(most_specific_types)


def get_graph_context(graph, target_class):
    if target_class not in graph:
        return {
            "parents": [],
            "children": [],
            "siblings": [],
        }

    parents = sorted(graph.successors(target_class))
    all_children = sorted(graph.predecessors(target_class))

    siblings = set()

    for parent in parents:
        siblings.update(graph.predecessors(parent))

    siblings.discard(target_class)

    return {
        "parents": parents,
        "children": all_children[:MAX_CHILDREN],
        "siblings": sorted(siblings)[:MAX_SIBLINGS],
    }


def has_usable_graph_context(context):
    return any(
        context[key]
        for key in ("parents", "children", "siblings")
    )


def select_demonstration_entity(entities, graph):
    entity = entities[0].copy()
    targets = find_most_specific_types(graph, entity)

    if len(targets) != 1:
        raise RuntimeError(
            "The first entity is not an unambiguous demonstration: "
            f"{targets}"
        )

    target_class = targets[0]
    context = get_graph_context(graph, target_class)

    if target_class not in graph or not has_usable_graph_context(context):
        raise RuntimeError(
            "The first entity has no usable graph context."
        )

    entity["target_class"] = target_class
    return entity, context


def select_test_entities(entities, graph, count, skip_count):
    selected = []
    usable_seen = 0

    skipped = {
        "multiple_or_no_targets": 0,
        "target_missing_from_graph": 0,
        "no_graph_context": 0,
        "usable_skipped_for_fresh_set": 0,
    }

    for entity in entities[1:]:
        targets = find_most_specific_types(graph, entity)

        if len(targets) != 1:
            skipped["multiple_or_no_targets"] += 1
            continue

        target_class = targets[0]

        if target_class not in graph:
            skipped["target_missing_from_graph"] += 1
            continue

        context = get_graph_context(graph, target_class)

        if not has_usable_graph_context(context):
            skipped["no_graph_context"] += 1
            continue

        if usable_seen < skip_count:
            usable_seen += 1
            skipped["usable_skipped_for_fresh_set"] += 1
            continue

        selected_entity = entity.copy()
        selected_entity["target_class"] = target_class

        selected.append(
            {
                "entity": selected_entity,
                "context": context,
            }
        )

        if len(selected) == count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f"Only {len(selected)} usable test entities were found; "
            f"{count} were requested."
        )

    return selected, skipped


# =========================================================
# 4. Load readable class labels
# =========================================================

def load_selected_labels(file_path, required_ids):
    labels = {}

    label_pattern = re.compile(
        r'^(wd:Q\d+)\s+rdfs:label\s+"((?:\\.|[^"])*)"\s+\.$'
    )

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = label_pattern.match(line.strip())

            if not match:
                continue

            class_id = match.group(1)
            class_label = match.group(2).replace(r"\"", '"')

            if class_id in required_ids:
                labels[class_id] = class_label

            if len(labels) == len(required_ids):
                break

    return labels


def readable_list(class_ids, labels):
    if not class_ids:
        return "None"

    return ", ".join(
        labels.get(class_id, class_id)
        for class_id in class_ids
    )


def ranked_gold_types(entity, labels):
    """
    Return all WiKC types for the entity, sorted from deepest/most
    specific to broadest. Duplicate class IDs are removed.
    """
    by_class_id = {}

    for item in entity["taxonomy_classes"]:
        class_id = item["class_id"]
        depth = item["depth"]

        if (
            class_id not in by_class_id
            or depth > by_class_id[class_id]
        ):
            by_class_id[class_id] = depth

    ranked = sorted(
        by_class_id.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )

    return [
        {
            "class_id": class_id,
            "label": labels.get(class_id, class_id),
            "depth": depth,
        }
        for class_id, depth in ranked
    ]


# =========================================================
# 5. Prompt construction
# =========================================================

COMMON_TASK = """
Predict up to 10 valid entity types for the entity.

Rank the types from the most specific and appropriate type to broader
but still valid types. Do not repeat a type. Use concise entity-type labels only. Do not include explanations,
descriptions, or entity names as types. If fewer than 10 valid
types can be inferred, return fewer than 10.
""".strip()

GRAPH_NOTE = """
The taxonomy context contains structural clues around an omitted target
type. It is not a list of answer options. The correct type may be absent
from the listed parent, sibling, and child classes.
""".strip()


def compact_json_types(labels):
    return json.dumps(
        {"types": labels},
        ensure_ascii=False,
    )


def create_prompts(
    example_entity,
    test_entity,
    example_context,
    test_context,
    labels,
):
    example_gold = ranked_gold_types(example_entity, labels)
    test_gold = ranked_gold_types(test_entity, labels)

    example_ranking = [
        item["label"]
        for item in example_gold[:TOP_K]
    ]

    if len(example_ranking) < TOP_K:
        raise RuntimeError(
            "The demonstration entity does not have 10 readable "
            "taxonomy types."
        )

    example_parents = readable_list(
        example_context["parents"],
        labels,
    )
    example_children = readable_list(
        example_context["children"],
        labels,
    )
    example_siblings = readable_list(
        example_context["siblings"],
        labels,
    )

    test_parents = readable_list(
        test_context["parents"],
        labels,
    )
    test_children = readable_list(
        test_context["children"],
        labels,
    )
    test_siblings = readable_list(
        test_context["siblings"],
        labels,
    )

    correct_explanation = (
        f'The description "{example_entity["description"]}" directly '
        f"supports {example_ranking[0]} as the most specific type. "
        "The remaining valid types are broader taxonomy classes, so "
        "they should follow in decreasing specificity."
    )

    wrong_ranking = [
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

    wrong_explanation = (
        "This ranking incorrectly treats the taxonomy neighbours as "
        "answer options, places the broader parent producer first, "
        "and omits the more specific type video game producer."
    )

    standard_prompt = f"""
{COMMON_TASK}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "...", "type 10"]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Answer:
{compact_json_types(example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Answer:
""".strip()

    graph_standard_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "...", "type 10"]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

Answer:
{compact_json_types(example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

Answer:
""".strip()

    cot_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

Provide a brief explanation, then return the ranked entity types as JSON
in exactly this form:
{{"types": ["type 1", "type 2", "...", "type 10"]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

Explanation:
{correct_explanation}

Answer:
{compact_json_types(example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

Explanation:

Answer:

""".strip()

    contrastive_cot_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

Provide a brief explanation, then return the ranked entity types as JSON
in exactly this form:
{{"types": ["type 1", "type 2", "...", "type 10"]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

Correct explanation:
{correct_explanation}

Correct ranking:
{compact_json_types(example_ranking)}

Wrong ranking:
{compact_json_types(wrong_ranking)}

Why the wrong ranking is wrong:
{wrong_explanation}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

Explanation:

Answer:

""".strip()

    prompts = {
        "standard": standard_prompt,
        "graph_aware_standard": graph_standard_prompt,
        "cot": cot_prompt,
        "contrastive_cot": contrastive_cot_prompt,
    }

    metadata = {
        "entity_id": test_entity["entity_id"],
        "entity_label": test_entity["entity_label"],
        "description": test_entity["description"],
        "reference_class_id": test_entity["target_class"],
        "reference_type": labels.get(
            test_entity["target_class"],
            test_entity["target_class"],
        ),
        "gold_types": test_gold,
        "graph_context": {
            "parents": test_parents,
            "siblings": test_siblings,
            "children": test_children,
        },
    }

    return prompts, metadata


# =========================================================
# 6. Main program
# =========================================================

def main():
    for file_path in (
        EVALUATION_FILE,
        TAXONOMY_FILE,
        LABELS_FILE,
    ):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    entities = read_entities(EVALUATION_FILE)
    graph = build_graph(TAXONOMY_FILE)

    example_entity, example_context = select_demonstration_entity(
        entities,
        graph,
    )

    selected_tests, skipped = select_test_entities(
        entities,
        graph,
        NUMBER_OF_TEST_ENTITIES,
        SKIP_USABLE_TEST_ENTITIES,
    )

    required_ids = {
        item["class_id"]
        for item in example_entity["taxonomy_classes"]
    }

    required_ids.update(
        {
            example_entity["target_class"],
            *example_context["parents"],
            *example_context["children"],
            *example_context["siblings"],
        }
    )

    for selected in selected_tests:
        test_entity = selected["entity"]
        test_context = selected["context"]

        required_ids.update(
            item["class_id"]
            for item in test_entity["taxonomy_classes"]
        )

        required_ids.update(
            {
                test_entity["target_class"],
                *test_context["parents"],
                *test_context["children"],
                *test_context["siblings"],
            }
        )

    labels = load_selected_labels(
        LABELS_FILE,
        required_ids,
    )

    OUTPUT.mkdir(exist_ok=True)
    manifest_rows = []

    for index, selected in enumerate(selected_tests, start=1):
        test_entity = selected["entity"]
        test_context = selected["context"]

        prompts, metadata = create_prompts(
            example_entity,
            test_entity,
            example_context,
            test_context,
            labels,
        )

        example_folder = OUTPUT / f"example_{index:02d}"
        example_folder.mkdir(exist_ok=True)

        metadata["example_number"] = index

        metadata_path = example_folder / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        for condition, prompt in prompts.items():
            prompt_path = (
                example_folder / f"{condition}_prompt.txt"
            )

            prompt_path.write_text(
                prompt,
                encoding="utf-8",
            )

            manifest_rows.append(
                {
                    "example_number": index,
                    "entity_id": metadata["entity_id"],
                    "entity_label": metadata["entity_label"],
                    "description": metadata["description"],
                    "condition": condition,
                    "prompt_path": str(prompt_path),
                    "metadata_path": str(metadata_path),
                }
            )

    manifest_path = OUTPUT / "manifest.csv"

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "example_number",
            "entity_id",
            "entity_label",
            "description",
            "condition",
            "prompt_path",
            "metadata_path",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Generated ranked prompts.\n")
    print("Demonstration entity:")
    print(" ", example_entity["entity_label"])

    print("\nSelected fresh test entities:")

    for index, selected in enumerate(selected_tests, start=1):
        entity = selected["entity"]
        reference_label = labels.get(
            entity["target_class"],
            entity["target_class"],
        )

        print(
            f"  {index:02d}. "
            f"{entity['entity_label']} -> {reference_label}"
        )

    print("\nSelection summary:")
    print(
        "  Previously used usable entities skipped:",
        skipped["usable_skipped_for_fresh_set"],
    )
    print(
        "  Multiple or no most-specific targets:",
        skipped["multiple_or_no_targets"],
    )
    print(
        "  Target absent from graph:",
        skipped["target_missing_from_graph"],
    )
    print(
        "  No graph context:",
        skipped["no_graph_context"],
    )

    print("\nTotal prompt files:", len(manifest_rows))
    print("Manifest:", manifest_path)
    print("Prompt folder:", OUTPUT)


if __name__ == "__main__":
    main()