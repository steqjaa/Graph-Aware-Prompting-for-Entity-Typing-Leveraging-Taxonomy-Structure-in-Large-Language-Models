from pathlib import Path
import csv
import json
import re

import networkx as nx


# =========================================================
# Configuration
# =========================================================

DATA = Path("data")
OUTPUT = Path("generated_20_prompts")

EVALUATION_FILE = DATA / "wikc_eval_1k.txt"
TAXONOMY_FILE = DATA / "wikc.tsv"
LABELS_FILE = DATA / "wiki_taxonomy_labels.tsv"

NUMBER_OF_TEST_ENTITIES = 20
MAX_SIBLINGS = 5
MAX_CHILDREN = 5


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
# 2. Build WiKC taxonomy graph
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
# 3. Identify usable target classes
# =========================================================

def find_most_specific_types(graph, entity):
    """
    Return classes for which no more-specific class belonging to the
    same entity can reach them through child -> parent edges.
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
    entity = entities[0]
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

    entity = entity.copy()
    entity["target_class"] = target_class

    return entity, context


def select_test_entities(entities, graph, count):
    selected = []
    skipped = {
        "multiple_or_no_targets": 0,
        "target_missing_from_graph": 0,
        "no_graph_context": 0,
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
# 4. Load labels
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


# =========================================================
# 5. Prompt construction
# =========================================================

NEIGHBOURHOOD_NOTE = (
    "The taxonomy context describes the direct neighbours of the "
    "unknown target type. The target type itself is intentionally "
    "omitted and may not appear among the listed classes. Do not "
    "assume that the answer must be one of the listed parent, "
    "sibling, or child classes."
)


def build_correct_explanation(
    entity,
    target_label,
    parent_text,
    sibling_text,
):
    parts = [
        f'The description states: "{entity["description"]}".',
        f"This directly supports the type {target_label}.",
    ]

    if parent_text != "None":
        parts.append(
            f"In the taxonomy, {target_label} is more specific than "
            f"its parent class or classes: {parent_text}."
        )

    if sibling_text != "None":
        parts.append(
            f"The sibling classes {sibling_text} are alternative "
            "types and do not match the description as precisely."
        )

    parts.append(
        f"Therefore, the most specific supported entity type is "
        f"{target_label}."
    )

    return " ".join(parts)


def create_prompts(
    example_entity,
    test_entity,
    example_context,
    test_context,
    labels,
):
    example_type = labels.get(
        example_entity["target_class"],
        example_entity["target_class"],
    )

    reference_type = labels.get(
        test_entity["target_class"],
        test_entity["target_class"],
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

    correct_explanation = build_correct_explanation(
        example_entity,
        example_type,
        example_parents,
        example_siblings,
    )

    if example_context["parents"]:
        wrong_type_id = example_context["parents"][0]
        wrong_type = labels.get(wrong_type_id, wrong_type_id)
    else:
        wrong_type = "entity"

    wrong_explanation = (
        f"{wrong_type} is listed as the direct parent class, so the "
        f"answer must be {wrong_type}."
    )

    why_wrong = (
        "The taxonomy context is not a list of answer options. It "
        "describes the neighbours of an omitted target class. "
        f"{wrong_type} is too broad, while the description directly "
        f"supports the more specific type {example_type}."
    )

    standard_prompt = f"""
Question:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

What is the most specific entity type?

Answer with only the entity type.

Answer: {example_type}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

What is the most specific entity type?

Answer with only the entity type.

Answer:
""".strip()

    graph_standard_prompt = f"""
Question:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

{NEIGHBOURHOOD_NOTE}

Infer the most specific entity type from the description and taxonomy
neighbourhood.

Answer with only the entity type.

Answer: {example_type}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

{NEIGHBOURHOOD_NOTE}

Infer the most specific entity type from the description and taxonomy
neighbourhood.

Answer with only the entity type.

Answer:
""".strip()

    cot_prompt = f"""
Question:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

{NEIGHBOURHOOD_NOTE}

What is the most specific entity type?

Explanation: {correct_explanation}

Answer: {example_type}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

{NEIGHBOURHOOD_NOTE}

What is the most specific entity type?

Provide a brief explanation followed by the entity type.

Explanation:
Answer:
""".strip()

    contrastive_cot_prompt = f"""
Question:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

{NEIGHBOURHOOD_NOTE}

What is the most specific entity type?

Correct explanation: {correct_explanation}

Correct answer: {example_type}

Wrong explanation: {wrong_explanation}

Why it is wrong: {why_wrong}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

{NEIGHBOURHOOD_NOTE}

What is the most specific entity type?

Provide a brief explanation followed by the entity type.

Explanation:
Answer:
""".strip()

    prompts = {
        "standard": standard_prompt,
        "graph_aware_standard": graph_standard_prompt,
        "cot": cot_prompt,
        "contrastive_cot": contrastive_cot_prompt,
    }

    details = {
        "reference_type": reference_type,
        "reference_class_id": test_entity["target_class"],
        "parents": test_parents,
        "siblings": test_siblings,
        "children": test_children,
    }

    return prompts, details


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
    )

    required_ids = {
        example_entity["target_class"],
        *example_context["parents"],
        *example_context["children"],
        *example_context["siblings"],
    }

    for selected in selected_tests:
        test_entity = selected["entity"]
        test_context = selected["context"]

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

        prompts, details = create_prompts(
            example_entity,
            test_entity,
            example_context,
            test_context,
            labels,
        )

        example_folder = OUTPUT / f"example_{index:02d}"
        example_folder.mkdir(exist_ok=True)

        metadata = {
            "example_number": index,
            "entity_id": test_entity["entity_id"],
            "entity_label": test_entity["entity_label"],
            "description": test_entity["description"],
            **details,
        }

        (example_folder / "metadata.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        for condition, prompt in prompts.items():
            prompt_path = example_folder / f"{condition}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            manifest_rows.append(
                {
                    "example_number": index,
                    "entity_id": test_entity["entity_id"],
                    "entity_label": test_entity["entity_label"],
                    "description": test_entity["description"],
                    "reference_class_id": details["reference_class_id"],
                    "reference_type": details["reference_type"],
                    "condition": condition,
                    "prompt_path": str(prompt_path),
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
            "reference_class_id",
            "reference_type",
            "condition",
            "prompt_path",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Generated prompts for 20 test entities.\n")
    print("Demonstration entity:")
    print(" ", example_entity["entity_label"])

    print("\nSelected test entities:")

    for index, selected in enumerate(selected_tests, start=1):
        test_entity = selected["entity"]
        reference_type = labels.get(
            test_entity["target_class"],
            test_entity["target_class"],
        )

        print(
            f"  {index:02d}. "
            f"{test_entity['entity_label']} -> {reference_type}"
        )

    print("\nSkipped while searching:")
    print(
        "  Multiple or no targets:",
        skipped["multiple_or_no_targets"],
    )
    print(
        "  Target missing from graph:",
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