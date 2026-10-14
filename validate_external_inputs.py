"""Verify frozen prompt/input hashes and full external-demonstration coverage."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parent
for name,expected in json.loads((ROOT/'frozen_inputs.json').read_text()).items():
 actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
 if actual!=expected:raise SystemExit('Frozen input/code changed: '+name)
ids=set();total=0
for part,expected in [('all_1k',9000),('children_removed',1353)]:
 rows=list(csv.DictReader((ROOT/f'prompts_v6/{part}/manifest.csv').open()))
 assert len(rows)==expected and len({(r['entity_id'],r['condition']) for r in rows})==expected
 for r in rows:
  m=json.loads((ROOT/r['metadata_path']).read_text());p=(ROOT/r['prompt_path']).read_text();head,task=p.split('Now classify:',1)
  assert m['entity_id']==r['entity_id'] and not m['demonstration_omitted']
  assert m['demonstration_entity_id']=='wd:Q18720219'
  assert head.count('Entity: Rod Fergusson')==1 and 'Katsuhiro Harada' not in head
  assert f"Entity: {m['entity_label']}" in task
  ids.add(r['entity_id']);total+=1
assert len(ids)==1000 and 'wd:Q18720219' not in ids and total==10353
print('Validated: 10,353 frozen prompts; external demonstration in every prompt; 1,000 evaluation entities.')
