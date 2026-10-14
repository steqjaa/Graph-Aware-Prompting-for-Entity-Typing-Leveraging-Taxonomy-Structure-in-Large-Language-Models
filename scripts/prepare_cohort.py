import argparse,csv,json,random,hashlib
from common import ROOT, ORIGINAL, CONDITIONS
import generate_ranked_prompts as base
import generate_graph_ablation_prompts as ablation
from compact_prompts import REPLACEMENTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--require',choices=['any','children','all'],default='children')
    p.add_argument('--neighbors',type=int,default=3,help='Maximum per relation, including parents; not a token budget.')
    p.add_argument('--exact',action='store_true',help='Require at least --neighbors in each relation before capping.')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--output',type=str,default='prompts_v4/fresh_children_100')
    a=p.parse_args()
    if a.count<1 or a.neighbors<1: p.error('count and neighbors must be positive')
    g=base.build_graph(ORIGINAL/'data/wikc.tsv')
    es=base.read_entities(ORIGINAL/'data/wikc_eval_1k.txt')
    demo, _=base.select_demonstration_entity(es,g)
    seen={demo['entity_id']}
    for f in ORIGINAL.glob('generated*prompts*/example*/metadata.json'):
        seen.add(json.loads(f.read_text())['entity_id'])
    # Legacy sets may also be represented only by saved results.
    for f in (ORIGINAL/'generated_ranked_results').glob('*/example*/*result.json'):
        eid=json.loads(f.read_text()).get('entity_id')
        if eid: seen.add(eid)
    def full_context(target):
        parents=sorted(g.successors(target)); children=sorted(g.predecessors(target))
        siblings=sorted(set().union(*(set(g.predecessors(x)) for x in parents))-{target})
        return dict(parents=parents,siblings=siblings,children=children)
    records=[];pool=[]
    for e in es:
        ts=base.find_most_specific_types(g,e)
        reason='';c={k:[] for k in ('parents','siblings','children')}
        if e['entity_id'] in seen: reason='previously_exposed_or_demo'
        elif len(ts)!=1:reason='multiple_or_no_most_specific_targets'
        elif ts[0] not in g:reason='target_absent_from_graph'
        else:
            c=full_context(ts[0])
            if not any(c.values()): reason='no_context'
        row=dict(entity_id=e['entity_id'],entity_label=e['entity_label'],exclusion=reason,**{k+'_count':len(v) for k,v in c.items()})
        records.append(row)
        if not reason:pool.append((e,ts[0],c))
    out=ROOT/a.output
    if out.exists() and any(out.iterdir()): raise SystemExit(f'Refusing to overwrite nonempty {out}; choose a new --output.')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'pool_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    eligible=[x for x in pool if (a.require=='any' or (bool(x[2]['children']) if a.require=='children' else all(x[2].values()))) and (not a.exact or all(len(v)>=a.neighbors for v in x[2].values()))]
    random.Random(a.seed).shuffle(eligible)
    summary=dict(seed=a.seed,total_entities=len(es),previously_exposed_ids=len(seen),unexposed_usable=len(pool),unexposed_with_children=sum(bool(x[2]['children']) for x in pool),unexposed_all_relations=sum(all(x[2].values()) for x in pool),eligible=len(eligible),requested=a.count,relation_filter=a.require,max_neighbors_per_relation=a.neighbors,exact_count_filter=a.exact,evaluation_status='NOT_RUN',context_regime='oracle',selection='seeded random sample from sorted input order after excluding all attached prior metadata/result IDs')
    (out/'cohort_design.json').write_text(json.dumps(summary,indent=2))
    if a.audit_only:print(json.dumps(summary,indent=2));return
    if len(eligible)<a.count:raise SystemExit(f'Only {len(eligible)} eligible; audit saved. Reduce count or relax filter.')
    selected=eligible[:a.count]
    # Deterministic, seeded per-node selection: relation sets are nested across budgets.
    def capped(target,c):
        result={}
        for k,v in c.items():
            vals=sorted(v);seed=int(hashlib.sha256(f'{a.seed}:{target}:{k}'.encode()).hexdigest()[:16],16)
            random.Random(seed).shuffle(vals);result[k]=vals[:a.neighbors]
        return result
    dc=capped(demo['target_class'],full_context(demo['target_class']))
    needed={x['class_id'] for x in demo['taxonomy_classes']}
    for vals in dc.values():needed.update(vals)
    for e,t,c in selected:
        needed.update(x['class_id'] for x in e['taxonomy_classes'])
        for vals in capped(t,c).values():needed.update(vals)
    labels=base.load_selected_labels(ORIGINAL/'data/wiki_taxonomy_labels.tsv',needed)
    for name,txt in REPLACEMENTS.items():setattr(base,name,txt)
    rows=[]
    for i,(e,t,context) in enumerate(selected,1):
        e=dict(e,target_class=t);c=capped(t,context)
        prompts,m=base.create_prompts(demo,e,dc,c,c,labels)
        prompts={k:v for k,v in prompts.items() if k in CONDITIONS}
        for relation,condition in [('parents','standard_parent_only'),('siblings','standard_siblings_only'),('children','standard_children_only')]:
            prompts[condition]=ablation.make_prompt(relation,demo,e,dc,c,labels)
        m.update(example_number=i,prompt_version='v4-compact-controlled-oracle',evaluation_status='NOT_RUN',context_ids=c,available_relation_counts={k:len(v) for k,v in context.items()},selected_relation_counts={k:len(v) for k,v in c.items()},cohort_seed=a.seed)
        # No shuffled condition is generated in this experiment.
        m.pop('shuffled_graph_context',None)
        m['context_method']='Oracle neighbourhood of the gold most-specific class; capped per relation.'
        folder=out/f'example_{i:03d}';folder.mkdir();mp=folder/'metadata.json';mp.write_text(json.dumps(m,indent=2,ensure_ascii=False))
        for condition,txt in prompts.items():
            pp=folder/f'{condition}_prompt.txt';pp.write_text(txt)
            rows.append(dict(example_number=i,entity_id=e['entity_id'],entity_label=e['entity_label'],description=e['description'],condition=condition,prompt_path=str(pp.relative_to(ROOT)),metadata_path=str(mp.relative_to(ROOT))))
    with (out/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2));print('Prompts:',len(rows),'Model calls: 0')
if __name__=='__main__':main()
