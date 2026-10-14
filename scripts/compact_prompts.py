"""
Method rationale: Wei et al. (2022), Chia et al. (2023), Shi et al. (2023).
"""
import csv, hashlib, json, shutil
from common import ROOT, ORIGINAL
import generate_ranked_prompts as old

REPLACEMENTS = {
'COMMON_TASK': 'List valid entity-type labels, most specific first, then broader valid types. Use concise labels, not entity names or descriptions. Include only evidence-supported types; do not repeat labels or pad the list.',
'GRAPH_NOTE': 'The context surrounds an omitted target type; it is evidence, not answer options. Parents are broader, children narrower, and siblings share a parent. Infer the type using the description and these relations. Copy a neighbour only if valid for this entity. Siblings need not be disjoint; predict a child only when the description independently supports it.',
'COT_NO_GRAPH_INSTRUCTIONS': 'Briefly explain the most specific supported type, reject unsupported nearby alternatives, and justify the broader types and their order. Use only the entity and description; do not assume supplied taxonomy evidence.',
'COT_GRAPH_INSTRUCTIONS': 'Briefly use parents to explain scope, compare sibling alternatives, and check children for unsupported specificity. Justify the final specific-to-broad order using the description and taxonomy evidence.',
'CONTRASTIVE_NO_GRAPH_INSTRUCTIONS': 'Contrast the best-supported type with plausible incorrect alternatives. Explain why unsupported, overly narrow, or incompatible alternatives are rejected.',
'CONTRASTIVE_GRAPH_INSTRUCTIONS': 'Contrast the inferred type with sibling and child alternatives. Reject an alternative only when the evidence warrants it.'
}

def compact(text):
    for name, replacement in REPLACEMENTS.items():
        text = text.replace(getattr(old, name), replacement)
    return text

def main():
    dest = ROOT / 'prompts_v4' / 'pilot_review'
    rows=[]; lengths=[]
    for folder in ['generated_ranked_prompts_v3', 'generated_graph_ablation_prompts_v1']:
        source=ORIGINAL/folder
        for file in sorted(source.glob('example*/*_prompt.txt')):
            target=dest/folder/file.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True)
            before=file.read_text(); after=compact(before); target.write_text(after)
            assert before.split('Example:',1)[-1] == after.split('Example:',1)[-1], 'Demonstration or task evidence changed'
            meta=json.loads((file.parent/'metadata.json').read_text())
            meta.update(prompt_version='v4-compact-review',evaluation_status='NOT_RUN',source_metadata=str((file.parent/'metadata.json').relative_to(ROOT)))
            mp=target.parent/'metadata.json'; mp.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
            condition=file.stem.removesuffix('_prompt')
            row=dict(example_number=int(file.parent.name.split('_')[1]),entity_id=meta['entity_id'],entity_label=meta['entity_label'],description=meta['description'],condition=condition,prompt_path=str(target.relative_to(ROOT)),metadata_path=str(mp.relative_to(ROOT)))
            rows.append(row)
            lengths.append(dict(source=folder,condition=condition,entity_id=meta['entity_id'],old_words=len(before.split()),new_words=len(after.split()),old_characters=len(before),new_characters=len(after),source_sha256=hashlib.sha256(before.encode()).hexdigest(),new_sha256=hashlib.sha256(after.encode()).hexdigest()))
    with (dest/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (ROOT/'analysis'/'prompt_length_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=lengths[0]);w.writeheader();w.writerows(lengths)
    (ROOT/'prompts_v4'/'instruction_blocks.json').write_text(json.dumps(REPLACEMENTS,indent=2))
    print(f'Wrote {len(rows)} shortened review prompts; model calls: 0.')
if __name__=='__main__': main()
