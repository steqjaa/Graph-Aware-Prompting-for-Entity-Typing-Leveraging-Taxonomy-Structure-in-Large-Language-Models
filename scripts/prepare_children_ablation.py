"""Create matched no-child prompts without regenerating any main prompts."""
import argparse
import csv
import hashlib
import json
from common import ROOT

CONDITIONS = ('graph_aware_standard', 'cot_graph', 'contrastive_cot_graph')

def remove_children(text):
    # Restrict instruction edits to the preamble and context edits to the test task.
    # The demonstration remains byte-for-byte identical, including its empty child context.
    head, task = text.split('Now classify:', 1)
    boundary = head.find('Example:')
    pre, demo = (head[:boundary], head[boundary:]) if boundary >= 0 else (head, '')
    pre = pre.replace('Parents are broader, children narrower, and siblings share a parent.',
                      'Parents are broader, and siblings share a parent.')
    pre = pre.replace(' Siblings need not be disjoint; predict a child only when the description independently supports it.',
                      ' Siblings need not be disjoint.')
    pre = pre.replace('compare sibling alternatives, and check children for unsupported specificity.',
                      'compare sibling alternatives, and reject unsupported specificity.')
    pre = pre.replace('sibling and child alternatives', 'sibling alternatives')
    lines = task.splitlines(keepends=True)
    removed = [line for line in lines if line.startswith('- Direct child classes:')]
    if len(removed) != 1:
        raise ValueError('Expected exactly one test-task child context line')
    return pre + demo + 'Now classify:' + ''.join(line for line in lines if line not in removed)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='prompts_v5/all_1k/manifest.csv')
    p.add_argument('--output', default='prompts_v5/children_removed')
    a = p.parse_args()
    out = ROOT / a.output
    if out.exists() and any(out.iterdir()):
        p.error('Output exists. Choose a new directory; never overwrite inferred prompts.')
    rows = list(csv.DictReader((ROOT / a.manifest).open()))
    selected = []
    entities = set()
    for row in rows:
        if row['condition'] not in CONDITIONS:
            continue
        meta = json.loads((ROOT / row['metadata_path']).read_text())
        if not meta['context_ids']['children']:
            continue
        text = (ROOT / row['prompt_path']).read_text()
        new = remove_children(text)
        entities.add(row['entity_id'])
        folder = out / f"example_{int(row['example_number']):04d}"
        folder.mkdir(parents=True, exist_ok=True)
        condition = row['condition'] + '_no_children'
        pp = folder / f'{condition}_prompt.txt'
        mp = folder / 'metadata.json'
        meta.update(source_context_ids=meta['context_ids'], source_condition=row['condition'],
                    children_removed=True, children_available=True,
                    prompt_version='v5-matched-children-removed', evaluation_status='NOT_RUN')
        # source_condition belongs to the manifest/result, not shared entity metadata.
        meta.pop('source_condition')
        meta['context_ids'] = dict(meta['context_ids'], children=[])
        for key in ('graph_context', 'oracle_graph_context'):
            meta[key] = dict(meta[key], children='None')
        pp.write_text(new)
        mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        selected.append(dict(row, condition=condition, prompt_path=str(pp.relative_to(ROOT)),
                             metadata_path=str(mp.relative_to(ROOT)), source_condition=row['condition'],
                             source_prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if not selected or len(selected) != 3 * len(entities):
        raise ValueError('Incomplete matched child cohort')
    with (out / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=selected[0]); w.writeheader(); w.writerows(selected)
    (out / 'design.json').write_text(json.dumps(dict(entities=len(entities), prompts=len(selected),
        comparison='Reuse main full-graph results for the same entity and source_condition',
        demonstration='Preserved verbatim', children_available='Original cohort availability'), indent=2))
    print(f'Child-present entities: {len(entities)}\nNew no-children prompts: {len(selected)}\nManifest: {out / "manifest.csv"}')

if __name__ == '__main__':
    main()
