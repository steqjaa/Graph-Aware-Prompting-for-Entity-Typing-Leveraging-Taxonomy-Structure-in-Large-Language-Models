from pathlib import Path
import csv
import re

import networkx as nx


# =========================================================
# Configuration
# =========================================================

DATA = Path("data")
OUTPUT = Path("generated_prompts")

EVALUATION_FILE = DATA / "wikc_eval_1k.txt"
TAXONOMY_FILE = DATA / "wikc.tsv"
LABELS_FILE = DATA / "wiki_taxonomy_labels.tsv"

# Prevent classes with hundreds of neighbours from creating huge prompts.
# These are temporary demo settings, not final thesis decisions.
MAX_SIBLINGS = 5
MAX_CHILDREN = 5


# =========================================================
# 1. Read the evaluation entities
# =========================================================

def read_entities(file_path):
    """
    Read all entities and their listed taxonomy classes from the
    WiKC evaluation file.

    Each row contains:
        entity ID
        entity label
        entity description
        class ID
        class depth
    """
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
                    f"Warning: skipped row {line_number} because "
                    f"depth is not an integer: {row}"
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

    if len(result) < 2:
        raise RuntimeError(
            "The evaluation file must contain at least two entities."
        )

    return result


# =========================================================
# 2. Build the WiKC graph
# =========================================================

def build_graph(file_path):
    """
    Build a directed graph whose edges are:

        child class -> parent class
    """
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
# 3. Identify most-specific classes
# =========================================================

def find_most_specific_types(graph, entity):
    """
    Find the most-specific classes listed for an entity.

    The graph stores edges as child -> parent. Therefore, a class is
    not most-specific when another class belonging to the same entity
    can reach it through one or more child-to-parent edges.

    Classes absent from wikc.tsv are handled safely. They are treated
    as having no known more-specific class in the available graph.
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
            # With child -> parent edges, NetworkX ancestors are the
            # more-specific nodes that can reach this class.
            more_specific_entity_types = (
                nx.ancestors(graph, class_id) & entity_type_ids
            )

        if not more_specific_entity_types:
            most_specific_types.append(class_id)

    return sorted(most_specific_types)


# =========================================================
# 4. Retrieve graph context
# =========================================================

def get_graph_context(graph, target_class):
    """
    Retrieve direct parents, direct children, and siblings.

    Since edges are child -> parent:
        successors(target)   = parents
        predecessors(target) = children
    """
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
    """Return True when at least one context category is non-empty."""
    return any(
        context[category]
        for category in ("parents", "children", "siblings")
    )


# =========================================================
# 5. Select demonstration and test entities
# =========================================================

def select_demonstration_entity(entities, graph):
    """
    Use the first evaluation entity as the demonstration entity.

    For the current WiKC file, this is Katsuhiro Harada. The function
    checks that it has exactly one most-specific class.
    """
    entity = entities[0]
    targets = find_most_specific_types(graph, entity)

    if len(targets) != 1:
        raise RuntimeError(
            "The first entity cannot be used as an unambiguous "
            f"demonstration. Candidates found: {targets}"
        )

    target_class = targets[0]

    if target_class not in graph:
        raise RuntimeError(
            "The demonstration target is absent from wikc.tsv, so "
            "graph context cannot be extracted."
        )

    context = get_graph_context(graph, target_class)

    if not has_usable_graph_context(context):
        raise RuntimeError(
            "The demonstration target has no usable graph context."
        )

    entity["target_class"] = target_class

    return entity, context


def select_test_entity(entities, graph):
    """
    Select the first test entity after the demonstration that:

    1. has exactly one most-specific class;
    2. has that class in wikc.tsv; and
    3. has usable graph context.

    This avoids arbitrarily forcing a multi-label entity such as
    Sony Alpha 230 into one target class.
    """
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

        entity["target_class"] = target_class

        return entity, context, skipped

    raise RuntimeError(
        "No unambiguous test entity with usable graph context was found."
    )


# =========================================================
# 6. Read only the labels that are needed
# =========================================================

def load_selected_labels(file_path, required_ids):
    """
    Scan the large RDF-style label file once and retain only the
    requested class labels.
    """
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
    """Convert class IDs into a comma-separated list of labels."""
    if not class_ids:
        return "None"

    return ", ".join(
        labels.get(class_id, class_id)
        for class_id in class_ids
    )


# =========================================================
# 7. Prompt construction
# =========================================================

def build_correct_explanation(
    entity,
    target_label,
    parent_text,
    sibling_text,
):
    """Build the demonstration explanation used in CoT prompts."""
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
    """Create the four agreed prompt conditions."""
    example_type = labels.get(
        example_entity["target_class"],
        example_entity["target_class"],
    )

    test_reference_type = labels.get(
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
        f"{example_entity['entity_label']} belongs to the broader "
        f"class {wrong_type}, so {wrong_type} is the appropriate "
        "entity type."
    )

    why_wrong = (
        f"{wrong_type} is a broader parent class and is therefore "
        "not the most specific type supported by the description. "
        f"The description directly supports {example_type}."
    )

    standard_prompt = f"""
Question:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

What is the most specific entity type?

Answer: {example_type}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

What is the most specific entity type?

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

What is the most specific entity type?

Answer: {example_type}

Question:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

What is the most specific entity type?

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

What is the most specific entity type?

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

What is the most specific entity type?

Explanation:
Answer:
""".strip()

    prompts = {
        "01_standard_prompt.txt": standard_prompt,
        "02_graph_aware_standard_prompt.txt": graph_standard_prompt,
        "03_cot_prompt.txt": cot_prompt,
        "04_contrastive_cot_prompt.txt": contrastive_cot_prompt,
    }

    prompt_details = {
        "example_type": example_type,
        "test_reference_type": test_reference_type,
        "example_parents": example_parents,
        "example_siblings": example_siblings,
        "example_children": example_children,
        "test_parents": test_parents,
        "test_siblings": test_siblings,
        "test_children": test_children,
    }

    return prompts, prompt_details


# =========================================================
# 8. Main program
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

    test_entity, test_context, skipped = select_test_entity(
        entities,
        graph,
    )

    required_ids = {
        example_entity["target_class"],
        test_entity["target_class"],
        *example_context["parents"],
        *example_context["children"],
        *example_context["siblings"],
        *test_context["parents"],
        *test_context["children"],
        *test_context["siblings"],
    }

    # Include the test entity's complete list of taxonomy classes
    # so that the diagnostic output can show readable labels.
    required_ids.update(
        item["class_id"]
        for item in test_entity["taxonomy_classes"]
    )

    labels = load_selected_labels(
        LABELS_FILE,
        required_ids,
    )

    prompts, details = create_prompts(
        example_entity,
        test_entity,
        example_context,
        test_context,
        labels,
    )

    OUTPUT.mkdir(exist_ok=True)

    for filename, prompt in prompts.items():
        output_file = OUTPUT / filename
        output_file.write_text(prompt, encoding="utf-8")

    print("Prompt generation complete.\n")

    print("Graph:")
    print("  Classes:", graph.number_of_nodes())
    print("  Edges:", graph.number_of_edges())

    print("\nDemonstration entity:")
    print(" ", example_entity["entity_label"])
    print("Demonstration type:")
    print(" ", details["example_type"])

    print("\nSelected test entity:")
    print(" ", test_entity["entity_label"])
    print("Description:")
    print(" ", test_entity["description"])

    print("\nTest graph context:")
    print("  Parents:", details["test_parents"])
    print("  Siblings:", details["test_siblings"])
    print("  Children:", details["test_children"])

    print(
        "\nReference type selected from an unambiguous "
        "most-specific candidate:"
    )
    print(" ", details["test_reference_type"])

    print("\nEntities skipped before selecting the test entity:")
    print(
        "  Multiple or no most-specific candidates:",
        skipped["multiple_or_no_targets"],
    )
    print(
        "  Target absent from wikc.tsv:",
        skipped["target_missing_from_graph"],
    )
    print(
        "  Target had no graph context:",
        skipped["no_graph_context"],
    )

    print("\nTaxonomy classes listed for the test entity:")

    sorted_test_classes = sorted(
        test_entity["taxonomy_classes"],
        key=lambda item: item["depth"],
        reverse=True,
    )

    for item in sorted_test_classes:
        class_id = item["class_id"]
        class_label = labels.get(class_id, "[label not found]")

        print(
            f"  Depth {item['depth']}: "
            f"{class_label} ({class_id})"
        )

    print("\nGenerated files:")

    for filename in prompts:
        print(" ", OUTPUT / filename)


if __name__ == "__main__":
    main()