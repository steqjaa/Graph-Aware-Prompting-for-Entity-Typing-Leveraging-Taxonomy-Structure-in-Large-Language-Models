from pathlib import Path
import argparse
import csv
import json
import re

import networkx as nx


# =========================================================
# Configuration
# =========================================================

DATA = Path("data")
DEFAULT_OUTPUT = Path("generated_ranked_prompts_v3")

EVALUATION_FILE = DATA / "wikc_eval_1k.txt"
TAXONOMY_FILE = DATA / "wikc.tsv"
LABELS_FILE = DATA / "wiki_taxonomy_labels.tsv"

DEFAULT_NUMBER_OF_TEST_ENTITIES = 20

DEFAULT_SKIP_USABLE_TEST_ENTITIES = 40

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

    if len(result) < 2:
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
Predict the valid entity types for the entity.

Rank the types from the most specific and appropriate type to broader
but still valid types. Do not repeat a type. Use concise entity-type labels
only. Do not include entity names or free-form descriptions as types.
Return only types that are supported by the entity description and the
available evidence. Do not force the list to a fixed length and do not
invent additional labels merely to make the list longer.
""".strip()

GRAPH_NOTE = """
The taxonomy context contains structural clues around an omitted target
type. It is not a list of answer options. Infer the missing type from the
entity description together with the structural position indicated by its
parent, sibling, and child classes. Parent classes are broader concepts,
child classes are narrower concepts, and sibling classes are nearby
alternatives under a shared parent. Do not copy a neighbouring class into
the answer unless it is itself a valid type for the entity. In particular,
do not predict a child class solely because it appears in the taxonomy
context; predict a narrower child only when the entity description itself
supports that narrower meaning.
""".strip()

COT_NO_GRAPH_INSTRUCTIONS = """
Before giving the final ranking, provide a concise explanation based only
on the entity and its description:
1. identify the most specific type directly supported by the description;
2. consider plausible nearby alternatives and reject alternatives that are
   unsupported or too specific;
3. identify broader valid types that still apply to the entity; and
4. explain why the returned types progress from more specific to broader.
Do not assume taxonomy information that is not present in the prompt.
""".strip()

COT_GRAPH_INSTRUCTIONS = """
Before giving the final ranking, provide a concise taxonomy-grounded
explanation. Explicitly use the graph structure:
1. explain what the parent class implies about the semantic scope of the
   omitted target type;
2. compare the entity with the sibling classes and identify any sibling
   that is clearly incompatible with the entity description;
3. use child classes to check whether a narrower subtype is independently
   supported by the entity description or would be too specific; and
4. explain why the returned types form a progression from more specific to
   broader valid classes.

Sibling classes are contrastive alternatives, not automatically logically
disjoint. Treat a sibling as incompatible only when the entity description
and taxonomy context justify that conclusion. Never infer a narrower child
class merely because it is listed in the graph context.
""".strip()

CONTRASTIVE_NO_GRAPH_INSTRUCTIONS = """
Use contrastive reasoning based only on the entity description. Compare the
best-supported type with plausible but incorrect alternatives. Explain why
unsupported, overly narrow, or semantically incompatible alternatives
should not be selected, then return the final ranking from specific to
broad. Do not assume taxonomy information that is not present in the prompt.
""".strip()

CONTRASTIVE_GRAPH_INSTRUCTIONS = """
Use contrastive taxonomy reasoning. Compare the inferred type against
nearby sibling and child alternatives, but remember that siblings are not
automatically disjoint and children are not automatically correct. Reject
an alternative only when the entity description and the structural evidence
justify doing so. A child class must be independently supported by the
entity description before it can be predicted.
""".strip()


def compact_json_types(labels):
    return json.dumps(
        {"types": labels},
        ensure_ascii=False,
    )


def compact_json_reasoned(explanation, labels):
    return json.dumps(
        {
            "explanation": explanation,
            "types": labels,
        },
        ensure_ascii=False,
    )


def create_prompts(
    example_entity,
    test_entity,
    example_context,
    test_context,
    shuffled_context,
    labels,
):
    example_gold = ranked_gold_types(example_entity, labels)
    test_gold = ranked_gold_types(test_entity, labels)

    example_ranking = [item["label"] for item in example_gold]

    example_parents = readable_list(example_context["parents"], labels)
    example_children = readable_list(example_context["children"], labels)
    example_siblings = readable_list(example_context["siblings"], labels)

    test_parents = readable_list(test_context["parents"], labels)
    test_children = readable_list(test_context["children"], labels)
    test_siblings = readable_list(test_context["siblings"], labels)

    shuffled_parents = readable_list(shuffled_context["parents"], labels)
    shuffled_children = readable_list(shuffled_context["children"], labels)
    shuffled_siblings = readable_list(shuffled_context["siblings"], labels)

    no_graph_explanation = (
        f'The description "{example_entity["description"]}" directly supports '
        f'{example_ranking[0]} as the most specific type. Plausible alternatives '
        'that require details absent from the description should be rejected. '
        'The remaining returned types are broader categories that still apply, '
        'so the ranking moves from specific to broad.'
    )

    graph_explanation = (
        f'The description "{example_entity["description"]}" supports '
        f'{example_ranking[0]} as the most specific type. In the taxonomy, '
        f'its direct parent is {example_parents}, which constrains the broader '
        'semantic category. The sibling classes are nearby alternatives, but '
        f'classes such as {example_siblings} are not as well supported by the '
        'description. The listed child context is '
        f'{example_children}; a child should be selected only if the description '
        'independently supports that narrower meaning. The remaining returned '
        'types are broader valid classes ranked in decreasing specificity.'
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

    wrong_no_graph_explanation = (
        "This ranking is wrong because it promotes unsupported alternatives, "
        "includes overly specific labels without evidence from the description, "
        "and fails to place the description-supported type first."
    )

    wrong_graph_explanation = (
        "This ranking is wrong because it treats taxonomy neighbours as answer "
        "options, promotes classes that the entity description does not support, "
        "and omits the more specific description-supported target type."
    )

    standard_prompt = f"""
{COMMON_TASK}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "..."]}}

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
{{"types": ["type 1", "type 2", "..."]}}

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

    shuffled_graph_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "..."]}}

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
- Direct parent classes: {shuffled_parents}
- Sibling classes: {shuffled_siblings}
- Direct child classes: {shuffled_children}

Answer:
""".strip()

    cot_no_graph_prompt = f"""
{COMMON_TASK}

{COT_NO_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "description-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Correct response:
{compact_json_reasoned(no_graph_explanation, example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Answer:
""".strip()

    cot_graph_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

{COT_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "taxonomy-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

Correct response:
{compact_json_reasoned(graph_explanation, example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

Answer:
""".strip()

    contrastive_no_graph_prompt = f"""
{COMMON_TASK}

{COT_NO_GRAPH_INSTRUCTIONS}

{CONTRASTIVE_NO_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "contrastive description-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Correct response:
{compact_json_reasoned(no_graph_explanation, example_ranking)}

Wrong ranking:
{compact_json_types(wrong_ranking)}

Why the wrong ranking is wrong:
{wrong_no_graph_explanation}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Answer:
""".strip()

    contrastive_graph_prompt = f"""
{COMMON_TASK}

{GRAPH_NOTE}

{COT_GRAPH_INSTRUCTIONS}

{CONTRASTIVE_GRAPH_INSTRUCTIONS}

Return JSON only in exactly this form:
{{"explanation": "contrastive taxonomy-grounded explanation", "types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- Direct parent classes: {example_parents}
- Sibling classes: {example_siblings}
- Direct child classes: {example_children}

Correct response:
{compact_json_reasoned(graph_explanation, example_ranking)}

Wrong ranking:
{compact_json_types(wrong_ranking)}

Why the wrong ranking is wrong:
{wrong_graph_explanation}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- Direct parent classes: {test_parents}
- Sibling classes: {test_siblings}
- Direct child classes: {test_children}

Answer:
""".strip()

    prompts = {
        "standard": standard_prompt,
        "graph_aware_standard": graph_standard_prompt,
        "shuffled_graph_standard": shuffled_graph_prompt,
        "cot_no_graph": cot_no_graph_prompt,
        "cot_graph": cot_graph_prompt,
        "contrastive_cot_no_graph": contrastive_no_graph_prompt,
        "contrastive_cot_graph": contrastive_graph_prompt,
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
        # Keep the historical key for compatibility with existing analysis.
        "graph_context": {
            "parents": test_parents,
            "siblings": test_siblings,
            "children": test_children,
        },
        "oracle_graph_context": {
            "parents": test_parents,
            "siblings": test_siblings,
            "children": test_children,
        },
        "shuffled_graph_context": {
            "parents": shuffled_parents,
            "siblings": shuffled_siblings,
            "children": shuffled_children,
        },
        "context_method": (
            "oracle context is retrieved around the hidden gold most-specific "
            "class; shuffled context is a deterministic negative control from "
            "another evaluation entity"
        ),
    }

    return prompts, metadata


# =========================================================
# 6. Main program
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a versioned ranked-prompt evaluation set."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_NUMBER_OF_TEST_ENTITIES,
        help="Number of usable test entities to generate.",
    )
    parser.add_argument(
        "--skip-usable",
        type=int,
        default=DEFAULT_SKIP_USABLE_TEST_ENTITIES,
        help=(
            "Number of usable entities to skip after the demonstration. "
            "Use 40 for the next clean set after the two earlier 20-entity sets."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output folder for prompts and metadata.",
    )
    args = parser.parse_args()
    output = args.output

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
        args.count,
        args.skip_usable,
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

    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for index, selected in enumerate(selected_tests, start=1):
        test_entity = selected["entity"]
        test_context = selected["context"]

        # Deterministic negative control: use the next test entity's local
        # taxonomy neighborhood, wrapping around at the end. The amount and
        # formatting of graph information stay the same, but the neighborhood
        # is unrelated to the current entity.
        shuffled_selected = selected_tests[index % len(selected_tests)]
        shuffled_context = shuffled_selected["context"]

        prompts, metadata = create_prompts(
            example_entity,
            test_entity,
            example_context,
            test_context,
            shuffled_context,
            labels,
        )

        example_folder = output / f"example_{index:02d}"
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

    manifest_path = output / "manifest.csv"

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
    print("Prompt folder:", output)


if __name__ == "__main__":
    main()