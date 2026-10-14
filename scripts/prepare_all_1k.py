"""Full exploratory 1k run: nine conditions; no retrieval or shuffled prompts.
Retains previously exposed entities. Oracle centre tie-breaking is explicit.
Method references: docs/references.md [1,5,6]; this full-cohort policy is new.
"""
import argparse,csv,json,random
from common import ROOT,ORIGINAL
import generate_ranked_prompts as base
import generate_graph_ablation_prompts as abl
from compact_prompts import REPLACEMENTS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='prompts_v5/all_1k');p.add_argument('--neighbors',type=int,default=3);a=p.parse_args()
 if a.neighbors<1:p.error('neighbors must be positive')
 out=ROOT/a.output
 if out.exists() and any(out.iterdir()):raise SystemExit('Output exists; choose a new --output (do not regenerate after inference).')
 es=base.read_entities(ORIGINAL/'data/wikc_eval_1k.txt');g=base.build_graph(ORIGINAL/'data/wikc.tsv')
 assert len(es)==1000,f'Expected 1000 entities, got {len(es)}'
 original_demo,dc=base.select_demonstration_entity(es,g)
 # Preserve the reviewed demonstration for 999 entities. For the original
 # demonstration entity, remove every example block instead of leaking its answer.
 demo=original_demo
 def context(t):
  if t not in g:return {k:[] for k in ['parents','siblings','children']}
  ps=sorted(g.successors(t));ss=sorted(set().union(*(set(g.predecessors(x)) for x in ps))-{t})
  return {k:v[:a.neighbors] for k,v in dict(parents=ps,siblings=ss,children=sorted(g.predecessors(t))).items()}
 dc=context(demo['target_class']);selected=[];needed={x['class_id'] for e in es for x in e['taxonomy_classes']}
 for e in es:
  targets=base.find_most_specific_types(g,e);depth={x['class_id']:x['depth'] for x in e['taxonomy_classes']}
  candidates=targets or sorted(depth)
  t=sorted(candidates,key=lambda x:(-depth.get(x,-1),x))[0]
  e=dict(e,target_class=t);c=context(t);selected.append((e,c,targets))
  for v in c.values():needed.update(v)
 for v in dc.values():needed.update(v)
 labels=base.load_selected_labels(ORIGINAL/'data/wiki_taxonomy_labels.tsv',needed)
 for n,v in REPLACEMENTS.items():setattr(base,n,v)
 out.mkdir(parents=True)
 rows=[];audits=[]
 def omit_demo(txt):
  before=txt.split('Example:',1)[0]
  # Contrastive prompts may use different positive-example headings.
  if 'Now classify:' not in txt:raise ValueError('Missing task boundary')
  import re
  # Everything after the response schema and before the target task is example content.
  lines=txt.splitlines();schema=next(i for i,l in enumerate(lines) if l.strip().startswith('{"') and '"types"' in l)
  return '\n'.join(lines[:schema+1])+'\n\nNow classify:'+txt.split('Now classify:',1)[1]
 for i,(e,c,targets) in enumerate(selected,1):
  prompts,m=base.create_prompts(demo,e,dc,c,c,labels)
  prompts.pop('shuffled_graph_standard',None)
  m.pop('shuffled_graph_context',None)
  m['context_method']='Oracle context around a deterministic gold class; no retrieved or shuffled conditions.'
  for rel,cond in [('parents','standard_parent_only'),('siblings','standard_siblings_only'),('children','standard_children_only')]:prompts[cond]=abl.make_prompt(rel,demo,e,dc,c,labels)
  is_demo=e['entity_id']==demo['entity_id']
  if is_demo:prompts={k:omit_demo(v) for k,v in prompts.items()}
  m.update(example_number=i,prompt_version='v5-full-cohort',evaluation_status='NOT_RUN',gold_types=base.ranked_gold_types(e,labels),oracle_centres_available=targets,oracle_centre_policy='deepest graph-minimal annotated class, class ID tie-break; all gold types retained for scoring',context_ids=c,children_available=bool(c['children']),demonstration_omitted=is_demo,previous_exposure_policy='all entities retained; exploratory re-evaluation')
  folder=out/f'example_{i:04d}';folder.mkdir();mp=folder/'metadata.json';mp.write_text(json.dumps(m,indent=2,ensure_ascii=False))
  for cond,txt in prompts.items():
   pp=folder/f'{cond}_prompt.txt';pp.write_text(txt)
   rows.append(dict(example_number=i,entity_id=e['entity_id'],entity_label=e['entity_label'],description=e['description'],condition=cond,prompt_path=str(pp.relative_to(ROOT)),metadata_path=str(mp.relative_to(ROOT))))
  audits.append(dict(entity_id=e['entity_id'],centres=len(targets),chosen_centre=e['target_class'],centre_in_graph=e['target_class'] in g,children_available=bool(c['children']),demonstration_omitted=is_demo))
 with (out/'manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (out/'cohort_audit.json').write_text(json.dumps(audits,indent=2))
 (out/'design.json').write_text(json.dumps(dict(entities=len(es),conditions=sorted({r['condition'] for r in rows}),prompts=len(rows),neighbors=a.neighbors,demonstration_policy='Original demo retained except for its own test case, which is zero-shot; report 999 paired few-shot cases separately.',limitations='Previously exposed entities included. Multi-target cases use one deterministic oracle centre. Empty contexts retained.'),indent=2))
 print(f'{len(es)} entities; {len(rows)} prompts. No Gemini calls made. Manifest: {out}/manifest.csv')
if __name__=='__main__':main()
