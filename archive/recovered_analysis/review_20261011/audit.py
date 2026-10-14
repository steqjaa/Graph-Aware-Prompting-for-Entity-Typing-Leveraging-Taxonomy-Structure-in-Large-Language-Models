import sys,json,glob,collections,hashlib,difflib
from pathlib import Path
import pandas as pd,numpy as np,re,csv
from functools import lru_cache
import argparse,os
parser=argparse.ArgumentParser(description="Audit archived thesis runs; no model inference.")
parser.add_argument("--project",required=True,type=Path,help="Extracted thesis-final project directory")
parser.add_argument("--output",default="audit-output",type=Path)
args=parser.parse_args()
ROOT=args.project.resolve()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);os.chdir(out)
Path("work").mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'scripts'))
def norm(t):
 return re.sub(r'\s+',' ',t.casefold().strip().replace('_',' ')).strip(" \t\n\r.,;:!?\"'")
def paired_summary(values):
 x=np.asarray(values);rng=np.random.default_rng(230917);v=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(2000)]);lo,hi=np.quantile(v,[.025,.975]);return dict(n=len(x),mean_delta=x.mean(),ci_low=lo,ci_high=hi,helped=int((x>1e-12).sum()),tied=int((abs(x)<=1e-12).sum()),harmed=int((x< -1e-12).sum()))
class Graph:
 def __init__(self):
  self.up=collections.defaultdict(set);self.down=collections.defaultdict(set);self.nodes=set();self.edges=set()
  for a,b in csv.reader(open(ROOT/'archive/original/data/wikc.tsv'),delimiter='\t'):
   self.up[a].add(b);self.down[b].add(a);self.nodes.update([a,b]);self.edges.add((a,b))
 def successors(self,x):return self.up[x]
 def predecessors(self,x):return self.down[x]
 def __len__(self):return len(self.nodes)
 def number_of_edges(self):return len(self.edges)
 @lru_cache(None)
 def ancestors(self,x):
  seen=set();todo=list(self.up[x])
  while todo:
   n=todo.pop()
   if n in seen:continue
   seen.add(n);todo.extend(self.up[n]-seen)
  return seen
g=Graph();T=g
metas=[json.load(open(f)) for f in sorted((ROOT/'prompts_v5/all_1k').glob('*/metadata.json'))]
rows=[]
for m in metas:
 gold={x['class_id'] for x in m['gold_types']};d={x['class_id']:x['depth'] for x in m['gold_types']};mx=max(d.values());gm={x for x in gold if d[x]==mx};c=m['reference_class_id'];ps=set(g.successors(c));ch=set(g.predecessors(c));ss=set().union(*(set(g.predecessors(p)) for p in ps))-{c}
 closed=gold|set().union(*(T.ancestors(x) for x in gold))
 row=dict(entity_id=m['entity_id'],entity_label=m['entity_label'],gold_count=len(gold),label_present=norm(m['reference_type']) in norm(m['description']),gold_not_closed=gold!=closed,gmax_tie=len(gm)>1,centre_not_gmax=c not in gm,description=m['description'],centre=m['reference_type'])
 for name,v in [('parents',ps),('siblings',ss),('children',ch)]:row[name+'_empty']=not v;row[name+'_truncated']=len(v)>3;row[name+'_count']=len(v)
 rows.append(row)
df=pd.DataFrame(rows);df.to_csv('work/entity_audit.csv',index=False)
summary={'classes':len(g),'edges':g.number_of_edges(),'entities':len(df),'mean_gold':df.gold_count.mean(),'counts':{x:int(df[x].sum()) for x in df if x.endswith(('_empty','_truncated')) or x in ['label_present','gold_not_closed','gmax_tie','centre_not_gmax']}}
print(json.dumps(summary,indent=2));json.dump(summary,open('work/stats.json','w'),indent=2)
allres={}; retries=[];thoughts=[];lengths=[];contrasts=[]
for model,path in [('Gemini','results/gemini_1k_fresh'),('Llama','results/json_mode_final/llama_1k_fast'),('Llama children','results/json_mode_final/llama_children_removed_fast')]:
 root=ROOT/path;final=[json.load(open(f)) for f in root.glob('example_*/*_result.json')];att=[json.load(open(f)) for f in (root/'attempts').glob('*.json')];allres[model]=final
 groups=collections.defaultdict(list)
 for r in att+final:
  groups[(r['entity_id'],r['condition'])].append(r)
 for cond in sorted({r['condition'] for r in final}):
  rs=[r for r in final if r['condition']==cond];lens=[len(r['predictions']) for r in rs]
  lengths.append(dict(model=model,condition=cond,n=len(rs),mean=np.mean(lens),median=np.median(lens)))
  fail=extra=maxa=0;st=collections.Counter()
  for r in rs:
   arr=groups[(r['entity_id'],cond)];unique={json.dumps({k:v for k,v in x.items() if k not in ['retry_provenance']},sort_keys=True):x for x in arr}
   # duplicate archived copies share timestamp and response; count inference calls, not files.
   uniq={ (x.get('timestamp_utc'),x.get('response'),x.get('error'),x.get('http_status')):x for x in unique.values()}
   seq=sorted(uniq.values(),key=lambda x:x.get('timestamp_utc',''))
   fail+=seq[0].get('status')!='completed';extra+=len(seq)-1;maxa=max(maxa,len(seq));st.update(x.get('status') for x in seq)
  retries.append(dict(model=model,condition=cond,n=len(rs),first_valid=len(rs)-fail,retried=fail,extra_calls=extra,max_attempts=maxa,statuses=dict(st)))
  if model=='Gemini':
   ts=[r.get('usage',{}).get('thoughtsTokenCount') for r in rs];tv=[x for x in ts if x is not None]
   thoughts.append(dict(condition=cond,n=len(rs),available=len(tv),total=sum(tv),mean=np.mean(tv),median=np.median(tv)))
 print(model,'DATES',min(x['timestamp_utc'] for x in final),max(x['timestamp_utc'] for x in final),'SEEDS',collections.Counter(x.get('generation_config',{}).get('seed') for x in final))
 if model!='Llama children':
  sc=pd.read_csv(root/'analysis/entity_scores.csv');sc=sc.merge(df[['entity_id','label_present','children_empty']],on='entity_id')
  for nongraph,graph,style in [('standard','graph_aware_standard','Standard'),('cot_no_graph','cot_graph','CoT'),('contrastive_cot_no_graph','contrastive_cot_graph','Contrastive')]:
   for present in [True,False]:
    a=sc[(sc.condition==nongraph)&(sc.label_present==present)].set_index('entity_id');b=sc[(sc.condition==graph)&(sc.label_present==present)].set_index('entity_id');s=paired_summary(b.f1-a.f1);contrasts.append(dict(model=model,style=style,group='Label present' if present else 'Label absent',**s))
  a=sc[(sc.condition=='standard')&sc.children_empty].set_index('entity_id');b=sc[(sc.condition=='standard_children_only')&sc.children_empty].set_index('entity_id');contrasts.append(dict(model=model,style='Children only versus Standard',group='No children',**paired_summary(b.f1-a.f1)))
for name,data in [('retries',retries),('thinking',thoughts),('lengths',lengths),('contrasts',contrasts)]:
 pd.DataFrame(data).to_csv('work/'+name+'.csv',index=False);json.dump(data,open('work/'+name+'.json','w'),indent=2);print(name);print(pd.DataFrame(data).to_string(index=False))
# selected illustrative entity with no literal centre label and useful parents
for m in metas:
 if m['entity_label']=='Sony Alpha 230':print('EXAMPLE',m)
# actual textual differences and full prompts for same illustrative entity
pp=ROOT/'prompts_v5/all_1k/example_0002'
for f in pp.glob('*prompt.txt'):
 (Path('work')/f.name).write_text(f.read_text())
a=(pp/'standard_prompt.txt').read_text().splitlines();b=(pp/'graph_aware_standard_prompt.txt').read_text().splitlines()
Path('work/prompt_diff.txt').write_text('\n'.join(difflib.unified_diff(a,b,fromfile='standard',tofile='graph_aware_standard',lineterm='')))
# recovery records for exact seed cases
print('SEED CASES',[(r['example_number'],r['condition'],r['generation_config']['seed']) for model,rs in allres.items() if model!='Gemini' for r in rs if r['generation_config']['seed']!=0])
