"""Recovered revision audits with explicit inputs."""
import json,collections,hashlib,difflib
from pathlib import Path
import pandas as pd
import numpy as np

def audit_extra(A, records, attempts, labels):
    # Hash equality and all metadata/prompt correspondence
    mismatch=[];metaerr=[];demo=collections.Counter();seeds=collections.Counter();Agroup=collections.defaultdict(list)
    for a in attempts:Agroup[a['_run'],int(a['example_number']),a['condition']].append(a)
    rows=[]
    for (run,i,c),r in records.items():
     demo[(r['metadata'].get('demonstration_entity_id'),r['metadata'].get('demonstration_omitted'))]+=1
     if not c.endswith('no_children') and run!='Gemini':
      b=records['Gemini',i,c]
      if b['prompt']!=r['prompt']:mismatch.append((run,i,c))
      if b['metadata']['gold_types']!=r['metadata']['gold_types']:metaerr.append((run,i,c))
     if run!='Gemini':seeds[(run,r['generation_config'].get('seed'))]+=1
     aa=Agroup[run,i,c]; allrec=aa+[r]; dedup={json.dumps({k:x.get(k) for k in ['timestamp_utc','status','response','input_hash']},sort_keys=True):x for x in allrec}; ordered=sorted(dedup.values(),key=lambda x:x.get('timestamp_utc','')); first=ordered[0]
     rows.append(dict(run=run,condition=c,example=i,attempts=len(ordered),retries=len(ordered)-1,first_valid=first.get('status')=='completed',final_valid=r.get('status')=='completed'))
    pd.DataFrame(rows).to_csv(A/'attempt_audit.csv',index=False)
    # Exact delta includes demonstration changes, not just graph note.
    b=records['Gemini',984,'standard']['prompt']; c=records['Gemini',984,'standard_children_only']['prompt'];
    (A/'empty_child_prompt_diff.txt').write_text('\n'.join(difflib.unified_diff(b.splitlines(),c.splitlines(),fromfile='Direct no graph',tofile='Children only')))
    # paired graph failure zero and worst-case delta bounds
    D=pd.read_csv(A/'entity_scores.csv');res=[]
    for run in D.run.unique():
     for a,b,sty in [('standard','graph_aware_standard','Direct'),('cot_no_graph','cot_graph','CoT'),('contrastive_cot_no_graph','contrastive_cot_graph','Contrastive')]:
      da=D[(D.run==run)&(D.condition==a)].set_index('example').f1;db=D[(D.run==run)&(D.condition==b)].set_index('example').f1
      res.append(dict(run=run,style=sty,zero_delta=(db.fillna(0)-da.fillna(0)).mean(),lower_bound=(db.fillna(0)-da.fillna(1)).mean(),upper_bound=(db.fillna(1)-da.fillna(0)).mean()))
    pd.DataFrame(res).to_csv(A/'failure_sensitivity.csv',index=False)
    json.dump({'prompt_mismatches':mismatch,'gold_mismatches':metaerr,'demo_counts':[(str(k),v) for k,v in demo.items()]},open(A/'alignment_audit.json','w'),indent=2)
