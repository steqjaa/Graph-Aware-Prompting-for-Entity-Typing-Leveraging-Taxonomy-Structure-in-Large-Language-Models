"""Final matched-cohort re-analysis, retaining the recovered revision's operation order.

The original recompute.py is preserved under archive/recovered_analysis.
Only storage/access and module boundaries changed for the core calculations.
"""
import json, collections, hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from . import metrics as old
from .io import load_records, validate_records


def recompute(root, out):
    out.mkdir(parents=True, exist_ok=True)
    records, attempts, configs = load_records(root)
    validation = validate_records(root, records)
    (out / 'completion_audit.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Verified 37,353 response records, including 53 unsuccessful outcomes.', flush=True)
    norm = old.normalize_label
    g = old.build_graph(root / 'archive/original/data/wikc.tsv')
    wanted = {norm(p) for r in records.values() for p in r.get('predictions', [])}
    labels = old.load_label_index(root / 'archive/original/data/wiki_taxonomy_labels.tsv', wanted=wanted)[0]
    @lru_cache(None)
    def anc(x):
        return frozenset(nx.descendants(g, x)) if x in g else frozenset()
    manifest = json.loads((root / 'configs/raw_archives.json').read_text())
    (out / 'run_configs.json').write_text(json.dumps(configs, indent=2) + '\n')
    meta={i:r['metadata'] for (run,i,c),r in records.items() if run=='Gemini'}
    assert len(meta)==1000
    entityrows=[]
    for i,m in sorted(meta.items()):
     gs=m['gold_types'];ids={v['class_id'] for v in gs};d=max(v['depth'] for v in gs);mx={v['class_id'] for v in gs if v['depth']==d};cc=m['reference_class_id'];ps=set(g.successors(cc)) if cc in g else set(); ch=set(g.predecessors(cc)) if cc in g else set();ss=set().union(*(set(g.predecessors(p)) for p in ps))-{cc}
     closure=ids|set().union(*(anc(x) for x in ids))
     row=dict(example=i,entity_id=m['entity_id'],entity=m['entity_label'],description=m['description'],centre=cc,centre_label=m['reference_type'],gold_count=len({norm(x['label']) for x in gs}),depth=d,multiple_deepest=len(mx)>1,centre_not_deepest=cc not in mx,not_closed=closure!=ids,missing_ancestors=len(closure-ids),label_in_description=norm(m['reference_type']) in norm(m['description']),children_available=bool(ch),parents=len(ps),siblings=len(ss),children=len(ch))
     for k,s in [('parents',ps),('siblings',ss),('children',ch)]:assert m['context_ids'][k]==sorted(s)[:3],(i,k)
     entityrows.append(row)
    E=pd.DataFrame(entityrows);E.to_csv(out/'entities.csv',index=False)
    rows=[];non_gold=[]
    for key,r in records.items():
     run,i,c=key;m=r['metadata'];gold=m['gold_types'];gs={norm(x['label']) for x in gold};ids={x['class_id'] for x in gold};up=ids|set().union(*(anc(x) for x in ids));pr=r.get('predictions',[]);valid=r.get('status')=='completed' and isinstance(pr,list) and bool(pr) and all(isinstance(x,str) and x.strip() for x in pr)
     if valid and 'cot' in c:valid=bool(r.get('explanation','').strip())
     row=dict(run=run,example=i,condition=c,valid=valid,status=r.get('status'),timestamp=r.get('timestamp_utc'),input_hash=r.get('input_hash'),predictions=json.dumps(pr,ensure_ascii=False),gold_count=len(gs),depth=max(x['depth'] for x in gold),thought_tokens=r.get('usage',{}).get('thoughtsTokenCount',r.get('raw_response',{}).get('usageMetadata',{}).get('thoughtsTokenCount',np.nan)),generated_tokens=r.get('generated_tokens',np.nan))
     if valid:
      ps=set(map(norm,pr));tp=len(ps&gs);mapped=set();opaque=set();counts=collections.Counter()
      for p in ps:
       exact=p in gs;cs={x['class_id'] for x in gold if norm(x['label'])==p} if exact else set(labels.get(p,set()))
       resolved=cs if exact or len(cs)==1 else set()
       mapped|=resolved
       if not resolved:opaque.add('unresolved:'+p)
       category='exact_gold' if exact else ('additional_ancestor' if resolved&up else 'other_taxonomy' if resolved else 'ambiguous' if cs else 'outside_label_inventory')
       counts[category]+=1
       if not exact: non_gold.append(dict(run=run,example=i,condition=c,label=p,category=category))
      pu=mapped|set().union(*(anc(x) for x in mapped))|opaque
      hf=2*len(pu&up)/(len(pu)+len(up))
      row.update(precision=tp/len(ps),recall=tp/len(gs),f1=2*tp/(len(ps)+len(gs)),mrr=old.reciprocal_rank(pr,gold,{}),ndcg=old.ndcg(pr,gold,{}),hf1=hf,predicted_count=len(ps),raw_count=len(pr),matched=tp,non_gold=len(ps)-tp,**{k:counts[k] for k in ['additional_ancestor','other_taxonomy','ambiguous','outside_label_inventory']})
     rows.append(row)
    D=pd.DataFrame(rows);D.to_csv(out/'entity_scores.csv',index=False);pd.DataFrame(non_gold).to_csv(out/'non_gold_labels.csv',index=False)
    metrics=['precision','recall','f1','mrr','ndcg','hf1','predicted_count','raw_count','non_gold','thought_tokens']
    S=D[D.valid].groupby(['run','condition'])[metrics].mean().reset_index();N=D.groupby(['run','condition']).agg(planned=('valid','size'),completed=('valid','sum')).reset_index();S=S.merge(N);S['failed']=S.planned-S.completed;S['failure_zero_f1']=S.f1*S.completed/S.planned;S.to_csv(out/'condition_summary.csv',index=False)
    rng=np.random.default_rng(230917)
    def compare(a,b,tag,**kw):
     j=a[['example']+metrics].merge(b[['example']+metrics],on='example',suffixes=('_a','_b'));j=j.dropna(subset=['f1_a','f1_b']);n=len(j)
     if n==0:return None
     x=(j.f1_b-j.f1_a).to_numpy(); boots=np.array([np.mean(x[rng.integers(n,size=n)]) for _ in range(2000)])
     rr=dict(kind=tag,n=n,delta=x.mean(),ci_low=np.quantile(boots,.025),ci_high=np.quantile(boots,.975),helped=int((x>1e-12).sum()),tied=int((abs(x)<=1e-12).sum()),harmed=int((x< -1e-12).sum()),**kw)
     for mt in metrics:rr['a_'+mt]=j[mt+'_a'].mean();rr['b_'+mt]=j[mt+'_b'].mean()
     return rr
    pairs=[('standard','graph_aware_standard','Direct'),('cot_no_graph','cot_graph','CoT'),('contrastive_cot_no_graph','contrastive_cot_graph','Contrastive')]
    C=[]
    for run in D.run.unique():
     for a,b,sty in pairs:
      sub=D[(D.run==run)&D.valid];aa=sub[sub.condition==a];bb=sub[sub.condition==b]
      C.append(compare(aa,bb,'graph',run=run,style=sty))
      for field,groups in [('label_in_description',[False,True]),('depth',['4–5','6–7','8+'])]:
       for v in groups:
        ids=E[E[field]==v].example if field!='depth' else E[E.depth.between(4,5) if v=='4–5' else E.depth.between(6,7) if v=='6–7' else E.depth>=8].example
        C.append(compare(aa[aa.example.isin(ids)],bb[bb.example.isin(ids)],field,run=run,style=sty,stratum=str(v)))
     for seed in ([] if run!='Gemini' else ['Llama 0','Llama 42','Llama 123']):
      for c in [p for a,b,s in pairs for p in [a,b]]+['standard_parent_only','standard_siblings_only','standard_children_only']:
       C.append(compare(D[(D.run==seed)&(D.condition==c)&D.valid],D[(D.run=='Gemini')&(D.condition==c)&D.valid],'model',run=seed,style=c))
     ids=E[~E.children_available].example
     C.append(compare(D[(D.run==run)&(D.condition=='standard')&D.example.isin(ids)&D.valid],D[(D.run==run)&(D.condition=='standard_children_only')&D.example.isin(ids)&D.valid],'empty_children',run=run,style='Direct'))
    for _,b,sty in pairs:
     no=b+'_no_children';ss=D[(D.run=='Llama 0')&D.valid]
     C.append(compare(ss[ss.condition==no],ss[ss.condition==b],'child',run='Llama 0',style=sty))
    C=pd.DataFrame([x for x in C if x]);C.to_csv(out/'comparisons.csv',index=False)
    # Same entities completed in all three seeds for each condition.
    seedrows=[]
    for c in S.condition.unique():
     if c.endswith('no_children'):continue
     pivot=D[(D.run!='Gemini')&(D.condition==c)&D.valid].pivot(index='example',columns='run',values='f1').dropna()
     means=pivot.mean();seedrows.append(dict(condition=c,n=len(pivot),**means.to_dict(),mean=means.mean(),sd=means.std(ddof=1),minimum=means.min(),maximum=means.max()))
    pd.DataFrame(seedrows).to_csv(out/'seed_summary.csv',index=False)
    # First attempts are provable for Llama because overwritten files are archived.
    A=collections.defaultdict(list)
    for a in attempts:
     if 'example_number' in a and 'condition' in a:A[(a['_run'],int(a['example_number']),a['condition'])].append(a)
    retry=[]
    for (run,i,c),r in records.items():
     aa=A[(run,i,c)]
     first=min(aa+[r],key=lambda a:a.get('timestamp_utc',''))
     retry.append(dict(run=run,condition=c,example=i,archived_attempts=len(aa),first_valid=(first.get('status')=='completed') if run!='Gemini' or aa else None,retried=bool(aa),final_valid=r['status']=='completed'))
    pd.DataFrame(retry).to_csv(out/'retries.csv',index=False)
    json.dump(manifest,open(out/'input_manifest.json','w'),indent=2)
    json.dump({'nodes':g.number_of_nodes(),'edges':g.number_of_edges(),'dag':nx.is_directed_acyclic_graph(g),'archive_attempt_records':len(attempts),'gemini_attempts':sum(a['_run']=='Gemini' for a in attempts),'dates':D.groupby('run').timestamp.agg(['min','max']).to_dict(),'gold_mean':E.gold_count.mean(),'not_closed':int(E.not_closed.sum()),'multiple_deepest':int(E.multiple_deepest.sum()),'centre_not_deepest':int(E.centre_not_deepest.sum()),'label_in_description':int(E.label_in_description.sum()),'empty':{k:int((E[k]==0).sum()) for k in ['parents','siblings','children']},'truncated':{k:int((E[k]>3).sum()) for k in ['parents','siblings','children']}},open(out/'audit.json','w'),indent=2)
    # Prompts + selected raw cases without opaque provider payload.
    for c in [a for a,b,s in pairs]+[b for a,b,s in pairs]+['standard_parent_only','standard_siblings_only','standard_children_only']:
     (out/(c+'_prompt.txt')).write_text(records[('Gemini',984,c)]['prompt'])
    cases=[]
    for key,r in records.items():
     if key[1] in [984,999,1,2,3]:cases.append({k:v for k,v in r.items() if k not in ['raw_response']}|{'run':key[0]})
    json.dump(cases,open(out/'cases.json','w'),indent=2,ensure_ascii=False)
    return D, E, S, C, records, attempts, labels, g
