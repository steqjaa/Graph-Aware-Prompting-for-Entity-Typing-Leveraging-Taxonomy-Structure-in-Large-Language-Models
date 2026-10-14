#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_TO_NEW = {
    "graph_aware_standard": "graph_aware_standard_no_children",
    "cot_graph": "cot_graph_no_children",
    "contrastive_cot_graph": "contrastive_cot_graph_no_children",
}

GRAPH_NOTE_WITH_CHILDREN = (
    "The context surrounds an omitted target type; it is evidence, not answer options. "
    "Parents are broader, children narrower, and siblings share a parent. Infer the type "
    "using the description and these relations. Copy a neighbour only if valid for this entity. "
    "Siblings need not be disjoint; predict a child only when the description independently supports it."
)
GRAPH_NOTE_NO_CHILDREN = (
    "The context surrounds an omitted target type; it is evidence, not answer options. "
    "Parents are broader and siblings share a parent. Infer the type using the description "
    "and these relations. Copy a neighbour only if valid for this entity. Siblings need not be disjoint."
)

COT_WITH_CHILDREN = (
    "Briefly use parents to explain scope, compare sibling alternatives, and check children for unsupported specificity. "
    "Justify the final specific-to-broad order using the description and taxonomy evidence."
)
COT_NO_CHILDREN = (
    "Briefly use parents to explain scope and compare sibling alternatives. "
    "Justify the final specific-to-broad order using the description and taxonomy evidence."
)

CONTRASTIVE_WITH_CHILDREN = (
    "Contrast the inferred type with sibling and child alternatives. Reject an alternative only when the evidence warrants it."
)
CONTRASTIVE_NO_CHILDREN = (
    "Contrast the inferred type with sibling alternatives. Reject an alternative only when the evidence warrants it."
)

DEMO_CHILD_SENTENCE = re.compile(
    r"\s*The listed child context is None; a child should be selected only if the description independently supports that narrower meaning\."
)
CHILD_CONTEXT_LINE = re.compile(r"^\s*- Direct child classes:.*(?:\n|$)", re.MULTILINE)


def remove_children(prompt: str) -> str:
    """Remove child-specific evidence/instructions while preserving everything else."""
    if GRAPH_NOTE_WITH_CHILDREN not in prompt:
        raise ValueError("Expected graph instruction block not found")
    prompt = prompt.replace(GRAPH_NOTE_WITH_CHILDREN, GRAPH_NOTE_NO_CHILDREN)
    prompt = prompt.replace(COT_WITH_CHILDREN, COT_NO_CHILDREN)
    prompt = prompt.replace(CONTRASTIVE_WITH_CHILDREN, CONTRASTIVE_NO_CHILDREN)
    prompt = DEMO_CHILD_SENTENCE.sub("", prompt)
    prompt = CHILD_CONTEXT_LINE.sub("", prompt)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    if "- Direct child classes:" in prompt:
        raise ValueError("Direct-child context line remained after transformation")
    return prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="prompts_v5/all_1k")
    ap.add_argument("--output", default="prompts_v5/children_removed")
    args = ap.parse_args()

    source = ROOT / args.source
    output = ROOT / args.output
    manifest_path = source / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output already exists and is non-empty: {output}")

    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    by_entity: dict[str, dict[str, dict[str, str]]] = {}
    for row in manifest:
        by_entity.setdefault(row["entity_id"], {})[row["condition"]] = row

    rows_out = []
    entity_audit = []
    selected = 0

    for entity_id, conditions in by_entity.items():
        first = next(iter(conditions.values()))
        meta = json.loads((ROOT / first["metadata_path"]).read_text(encoding="utf-8"))
        if not bool(meta.get("children_available")):
            continue

        child_ids = meta.get("context_ids", {}).get("children", [])
        if not child_ids:
            raise ValueError(f"children_available=True but no child IDs for {entity_id}")

        selected += 1
        example_number = int(first["example_number"])
        folder = output / f"example_{example_number:04d}"
        folder.mkdir(parents=True, exist_ok=True)

        new_meta = dict(meta)
        new_meta.update(
            child_removal_ablation=True,
            child_removal_source="prompts_v5/all_1k",
            children_removed=True,
            original_child_ids=child_ids,
            ablation_context_ids={
                "parents": meta.get("context_ids", {}).get("parents", []),
                "siblings": meta.get("context_ids", {}).get("siblings", []),
                "children": [],
            },
        )
        mp = folder / "metadata.json"
        mp.write_text(json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        for source_condition, new_condition in SOURCE_TO_NEW.items():
            row = conditions.get(source_condition)
            if row is None:
                raise ValueError(f"Missing {source_condition} for {entity_id}")
            source_prompt = (ROOT / row["prompt_path"]).read_text(encoding="utf-8")
            new_prompt = remove_children(source_prompt)
            pp = folder / f"{new_condition}_prompt.txt"
            pp.write_text(new_prompt, encoding="utf-8")
            rows_out.append({
                "example_number": example_number,
                "entity_id": row["entity_id"],
                "entity_label": row["entity_label"],
                "description": row["description"],
                "condition": new_condition,
                "prompt_path": str(pp.relative_to(ROOT)),
                "metadata_path": str(mp.relative_to(ROOT)),
                "source_with_children_condition": source_condition,
            })

        entity_audit.append({
            "example_number": example_number,
            "entity_id": entity_id,
            "entity_label": first["entity_label"],
            "children_count": len(child_ids),
            "children_ids": child_ids,
        })

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    (output / "cohort_audit.json").write_text(
        json.dumps(entity_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "design.json").write_text(
        json.dumps({
            "design": "matched child-removal ablation",
            "source_prompt_set": args.source,
            "entities": selected,
            "new_conditions": list(SOURCE_TO_NEW.values()),
            "new_prompts": len(rows_out),
            "with_children_results_to_reuse": list(SOURCE_TO_NEW.keys()),
            "estimand": (
                "Among entities with at least one direct child in the capped oracle context, "
                "compare the same entity and prompting family with parents+sibling+children "
                "against parents+sibling context with child evidence/instructions removed."
            ),
            "note": "No with-children prompts are duplicated; reuse the main 1k run.",
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Child-present entities: {selected}")
    print(f"New no-children prompts: {len(rows_out)}")
    print(f"Manifest: {output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
