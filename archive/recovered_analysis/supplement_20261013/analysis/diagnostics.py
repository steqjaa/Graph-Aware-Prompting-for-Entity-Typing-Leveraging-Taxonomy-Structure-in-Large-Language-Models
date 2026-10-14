import pickle,json,collections,re,string,hashlib,os
import pandas as pd,numpy as np
D,A,C=pickle.load(open('analysis/records.pkl','rb'))
def norm(s):return re.sub(r'\s+',' ',s.lower().replace('_',' ')).strip().strip(string.punctuation+' ').strip()
meta={i:j['metadata'] for (i,c),j in D['Gemini'].items() if c=='standard'}
issues=[];dup=collections.Counter();dates={};prompt_hashes={}
for g in D:
 for key,j in D[g].items():
  if j['status']!='completed':continue
  pairs=[]
  def hook(ps):
   ks=[k for k,v in ps]
   if len(ks)!=len(set(ks)):pairs.append(1)
   return dict(ps)
  try:
   x=json.loads(j['response'],object_pairs_hook=hook)
   if x.get('types')!=j['predictions']:issues.append((g,key,'pred mismatch'))
  except Exception as e:issues.append((g,key,'not bare JSON'))
  if pairs:dup[g]+=1
  if j['finish_reason']!='STOP':issues.append((g,key,'finish'))
print('VALIDATION',len(issues),collections.Counter(x[2] for x in issues),'DUP',dup)
# source gold alignment
source=collections.defaultdict(set)
for l in open('analysis/source/wikc_eval_1k.txt'):
 parts=l.rstrip().split('\t');source[parts[0]].add((parts[-2],int(parts[-1])))
print('SOURCE GOLD MISMATCH',sum(source[m['entity_id']]!={(x['class_id'],x['depth']) for x in m['gold_types']} for m in meta.values()))
# topological DAG and metadata anomalies
up=collections.defaultdict(set);down=collections.defaultdict(set)
for l in open('analysis/source/wikc.tsv'):
 a,b=l.strip().split('\t');up[a].add(b);down[b].add(a)
nodes=set(up)|set(down);degrees={q:len(up[q]) for q in nodes};todo=[q for q,d in degrees.items() if d==0];visited=0
while todo:
 q=todo.pop();visited+=1
 for child in down[q]:
  degrees[child]-=1
  if degrees[child]==0:todo.append(child)
print('ACYCLIC',visited==len(nodes))
print('GOLD ABSENT',[(i,m['entity_label'],[x['class_id'] for x in m['gold_types'] if x['class_id'] not in nodes]) for i,m in meta.items() if any(x['class_id'] not in nodes for x in m['gold_types'])])
# exact control difference
import difflib
for c in ['graph_aware_standard','standard_children_only','cot_graph','contrastive_cot_graph']:
 b={'cot_graph':'cot_no_graph','contrastive_cot_graph':'contrastive_cot_no_graph'}.get(c,'standard')
 diff='\n'.join(difflib.unified_diff(D['Gemini'][(1,b)]['prompt'].splitlines(),D['Gemini'][(1,c)]['prompt'].splitlines(),fromfile=b,tofile=c));open('output/prompt_diff_'+c+'.txt','w').write(diff)
print('EMPTY CONTROL\n',open('output/prompt_diff_standard_children_only.txt').read())
# detailed output inspection three entities for all conditions
out=[]
for i in [1,2,5]:
 for g in D:
  for (ii,c),j in sorted(D[g].items()):
   if ii==i and 'no_children' not in c:out.append({k:j.get(k) for k in ['entity_label','description','condition','predictions','explanation','status']}|{'model':g,'entity':i,'gold':[x['label'] for x in meta[i]['gold_types']]})
json.dump(out,open('output/qualitative_outputs.json','w'),indent=2,ensure_ascii=False)
# per condition coverage
s=pd.read_csv('output/entity_scores.csv');co=s.groupby(['model','condition'])[['gold','ancestor_outside_gold','taxonomy_non_gold','unresolved','n_types','n_gold','tp']].sum();co['micro_gold_precision']=co.tp/co.n_types;co['micro_gold_recall']=co.tp/co.n_gold;co.to_csv('output/coverage_summary.csv')
rs=pd.read_csv('output/retry_audit.csv');rs.groupby(['model','condition']).agg(first_valid=('first_valid','sum'),retries=('retries','sum'),max_attempts=('attempts','max'),completed=('final_valid','sum'),n=('entity','size')).to_csv('output/retry_summary.csv')
seeds=s[s.model!='Gemini'];seeds=seeds[~seeds.condition.str.contains('no_children')];agg=seeds.groupby(['model','condition']).f1.mean().reset_index();agg.groupby('condition').f1.agg(['mean','std','min','max']).to_csv('output/seed_summary.csv')
for c in ['standard','graph_aware_standard','cot_no_graph','cot_graph','contrastive_cot_no_graph','contrastive_cot_graph']:
 print('COVERAGE',c,co.loc[('Gemini',c)].to_dict())
print('SEED\n',pd.read_csv('output/seed_summary.csv').to_string(index=False))
json.dump({'validation_issues':issues,'duplicate_keys':dict(dup),'taxonomy_dag':visited==len(nodes),'source_sha256':{f:hashlib.sha256(open('analysis/source/'+f,'rb').read()).hexdigest() for f in ['wikc.tsv','wikc_eval_1k.txt','dataset.ipynb']},'archive_sha256':{f:hashlib.sha256(open('upload/'+f,'rb').read()).hexdigest() for f in os.listdir('upload') if f.endswith('.zip')}},open('output/provenance.json','w'),indent=2)
# save compact data needed for writing
json.dump(meta,open('analysis/meta.json','w'))
