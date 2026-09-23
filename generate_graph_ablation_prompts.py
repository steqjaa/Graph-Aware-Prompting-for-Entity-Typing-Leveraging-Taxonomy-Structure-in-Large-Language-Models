#!/usr/bin/env python3
"""
Generate ONLY the three new Standard-prompt graph ablation conditions:

    standard_parent_only
    standard_siblings_only
    standard_children_only

The existing results for:
    standard
    graph_aware_standard
are intentionally reused, so for 20 entities this creates only 60 new prompts.

Run from the thesis project root, beside generate_ranked_prompts.py.
"""

from pathlib import Path
import argparse
import csv
import json

import generate_ranked_prompts as base


DEFAULT_OUTPUT = Path("generated_graph_ablation_prompts_v1")
DEFAULT_COUNT = 20
DEFAULT_SKIP = 40


RELATION_INFO = {
    "parents": {
        "condition": "standard_parent_only",
        "display": "Direct parent classes",
        "note": (
            "The taxonomy context contains only direct parent classes of an omitted "
            "target type. Parent classes are broader than the omitted target and can "
            "constrain its semantic scope. They are structural clues, not answer "
            "options. Do not copy a parent into the answer unless it is independently "
            "a valid type for the entity."
        ),
    },
    "siblings": {
        "condition": "standard_siblings_only",
        "display": "Sibling classes",
        "note": (
            "The taxonomy context contains only sibling classes of an omitted target "
            "type. Siblings share a direct parent with the omitted target and are "
            "nearby alternatives, but they are not necessarily logically disjoint. "
            "They are structural clues, not answer options. Do not copy a sibling "
            "into the answer unless it is independently a valid type for the entity."
        ),
    },
    "children": {
        "condition": "standard_children_only",
        "display": "Direct child classes",
        "note": (
            "The taxonomy context contains only direct child classes of an omitted "
            "target type. Child classes are narrower than the omitted target and can "
            "help indicate its specificity boundary. They are structural clues, not "
            "answer options. Do not predict a child merely because it is listed; use "
            "a narrower child only when the entity description independently supports it."
        ),
    },
}


def make_prompt(
    relation_key,
    example_entity,
    test_entity,
    example_context,
    test_context,
    labels,
):
    info = RELATION_INFO[relation_key]
    example_gold = base.ranked_gold_types(example_entity, labels)
    example_ranking = [item["label"] for item in example_gold]

    example_values = base.readable_list(example_context[relation_key], labels)
    test_values = base.readable_list(test_context[relation_key], labels)

    return f"""
{base.COMMON_TASK}

{info["note"]}

Return JSON only in exactly this form:
{{"types": ["type 1", "type 2", "..."]}}

Example:

Entity: {example_entity["entity_label"]}
Description: {example_entity["description"]}

Taxonomy context:
- {info["display"]}: {example_values}

Answer:
{base.compact_json_types(example_ranking)}

Now classify:

Entity: {test_entity["entity_label"]}
Description: {test_entity["description"]}

Taxonomy context:
- {info["display"]}: {test_values}

Answer:
""".strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate parent/sibling/child-only graph ablation prompts."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--skip-usable", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (
        base.EVALUATION_FILE,
        base.TAXONOMY_FILE,
        base.LABELS_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    entities = base.read_entities(base.EVALUATION_FILE)
    graph = base.build_graph(base.TAXONOMY_FILE)

    example_entity, example_context = base.select_demonstration_entity(
        entities, graph
    )
    selected_tests, skipped = base.select_test_entities(
        entities,
        graph,
        args.count,
        args.skip_usable,
    )

    # Load labels needed for gold classes and every ablation context.
    required_ids = {
        item["class_id"]
        for item in example_entity["taxonomy_classes"]
    }
    required_ids.add(example_entity["target_class"])
    for relation in ("parents", "siblings", "children"):
        required_ids.update(example_context[relation])

    for selected in selected_tests:
        entity = selected["entity"]
        context = selected["context"]
        required_ids.update(
            item["class_id"] for item in entity["taxonomy_classes"]
        )
        required_ids.add(entity["target_class"])
        for relation in ("parents", "siblings", "children"):
            required_ids.update(context[relation])

    labels = base.load_selected_labels(base.LABELS_FILE, required_ids)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    availability = {
        "parents": {"non_empty": 0, "empty_entities": []},
        "siblings": {"non_empty": 0, "empty_entities": []},
        "children": {"non_empty": 0, "empty_entities": []},
    }

    for index, selected in enumerate(selected_tests, start=1):
        entity = selected["entity"]
        context = selected["context"]
        gold_types = base.ranked_gold_types(entity, labels)

        example_folder = args.output / f"example_{index:02d}"
        example_folder.mkdir(exist_ok=True)

        relation_availability = {}
        relation_counts = {}

        for relation in ("parents", "siblings", "children"):
            has_values = bool(context[relation])
            relation_availability[relation] = has_values
            relation_counts[relation] = len(context[relation])

            if has_values:
                availability[relation]["non_empty"] += 1
            else:
                availability[relation]["empty_entities"].append(
                    entity["entity_label"]
                )

        metadata = {
            "example_number": index,
            "entity_id": entity["entity_id"],
            "entity_label": entity["entity_label"],
            "description": entity["description"],
            "reference_class_id": entity["target_class"],
            "reference_type": labels.get(
                entity["target_class"], entity["target_class"]
            ),
            "gold_types": gold_types,
            "oracle_graph_context": {
                "parents": base.readable_list(context["parents"], labels),
                "siblings": base.readable_list(context["siblings"], labels),
                "children": base.readable_list(context["children"], labels),
            },
            "relation_available": relation_availability,
            "relation_counts": relation_counts,
            "ablation_design": (
                "oracle local taxonomy context around the hidden most-specific "
                "gold class; target class itself is omitted; exactly one relation "
                "type is shown in each prompt"
            ),
        }

        metadata_path = example_folder / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for relation in ("parents", "siblings", "children"):
            condition = RELATION_INFO[relation]["condition"]
            prompt = make_prompt(
                relation,
                example_entity,
                entity,
                example_context,
                context,
                labels,
            )

            prompt_path = example_folder / f"{condition}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            manifest_rows.append({
                "example_number": index,
                "entity_id": entity["entity_id"],
                "entity_label": entity["entity_label"],
                "description": entity["description"],
                "condition": condition,
                "prompt_path": str(prompt_path),
                "metadata_path": str(metadata_path),
            })

    manifest_path = args.output / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        fields = [
            "example_number",
            "entity_id",
            "entity_label",
            "description",
            "condition",
            "prompt_path",
            "metadata_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    design = {
        "purpose": "local graph relation ablation",
        "reused_existing_conditions": [
            "standard",
            "graph_aware_standard",
        ],
        "new_conditions": [
            "standard_parent_only",
            "standard_siblings_only",
            "standard_children_only",
        ],
        "new_prompt_count": len(manifest_rows),
        "entity_count": len(selected_tests),
        "skip_usable": args.skip_usable,
        "relation_availability": availability,
        "primary_analysis": {
            "all_entities": (
                "Treat an empty relation list as real structural information "
                "(e.g. no direct children in WiKC)."
            ),
            "non_empty_sensitivity_analysis": (
                "Also report parent/sibling/child effects only among entities "
                "where that relation contains at least one class."
            ),
        },
    }

    (args.output / "ablation_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Generated graph ablation prompts.")
    print(f"Entities: {len(selected_tests)}")
    print(f"New prompt files: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Relation availability:")
    for relation in ("parents", "siblings", "children"):
        data = availability[relation]
        print(
            f"  {relation:10s}: "
            f"{data['non_empty']}/{len(selected_tests)} non-empty"
        )
    print()
    print("Expected manifest line count including header:", len(manifest_rows) + 1)


if __name__ == "__main__":
    main()