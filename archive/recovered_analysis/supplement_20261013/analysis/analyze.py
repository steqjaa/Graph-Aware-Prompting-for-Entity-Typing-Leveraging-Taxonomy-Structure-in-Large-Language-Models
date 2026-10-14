import zipfile,json,re,os,collections,hashlib,math,pickle,string
import numpy as np,pandas as pd
from functools import lru_cache
os.makedirs('output',exist_ok=True)
D={}; attempts={}; configs={}
for group,fn in [('Gemini','gemini.zip'),('Llama 0','llama_seed_0_results.zip'),('Llama 42','llama_seed_42_results.zip'),('Llama 123','llama_seed_123_results.zip')]:
 z=zipfile.ZipFile('upload/'+fn);D[group]={};attempts[group]=[];configs[group]={}
 for n in z.namelist():
  if not n.endswith('.json') or '__MACOSX' in n or '/._' in n:continue
  j=json.loads(z.read(n))
  if '/attempts/' in n:attempts[group].append(j)
  elif re.search(r'/example_\d+/[^/]+_result.json$',n):
   key=(int(j['example_number']),j['condition']);assert key not in D[group];D[group][key]=j
  elif '/example_' not in n:configs[group][n]=j
pickle.dump((D,attempts,configs),open('analysis/records.pkl','wb'))
print('records', {k:len(v) for k,v in D.items()},'attempts',{k:len(v) for k,v in attempts.items()})
for g in D:print(g,collections.Counter(j['status'] for j in D[g].values()),'attempt statuses',collections.Counter(j['status'] for j in attempts[g]))
json.dump(configs,open('analysis/configs.json','w'),indent=2)
def norm(s):return re.sub(r'\s+',' ',s.lower().replace('_',' ')).strip().strip(string.punctuation+' ').strip()
meta={i:j['metadata'] for (i,c),j in D['Gemini'].items() if c=='standard'}
assert len(meta)==1000
up=collections.defaultdict(set);down=collections.defaultdict(set);nodes=set()
for l in open('analysis/source/wikc.tsv'):
 a,b=l.strip().split('\t');up[a].add(b);down[b].add(a);nodes.update([a,b])
@lru_cache(None)
def anc(n):
 out=set();todo=list(up[n])
 while todo:
  a=todo.pop()
  if a not in out:out.add(a);todo.extend(up[a])
 return frozenset(out)
labels={}
for m in meta.values():
 for x in m['gold_types']:labels[x['class_id']]=x['label']
 for r,ids in m['context_ids'].items():
  ls=m['oracle_graph_context'][r].split(', ')
  if len(ls)==len(ids):labels.update(zip(ids,ls))
# full label file parsed if present, retaining only taxonomy nodes
p='analysis/source/wiki_taxonomy_labels_full.tsv'
if os.path.exists(p):
 for l in open(p):
  row=l.rstrip('\n').split('\t')
  if row[0] in nodes:
   val=row[2].strip(chr(34))
   if val.endswith('@en'):val=val[:-3].strip('"')
   labels[row[0]]=val
labelmap=collections.defaultdict(set)
for q,l in labels.items():
 if q in nodes:labelmap[norm(l)].add(q)
print('nodes labels',len(nodes),len(labels),list(labels.items())[:3])
stats={'classes':len(nodes),'edges':sum(map(len,up.values())),'entities':1000,'mean_gold':float(np.mean([len(m['gold_types']) for m in meta.values()])),'not_closed':0,'max_ties':0,'centre_not_max':0,'centre_mismatch':0,'context_mismatch':0,'description_label':0,'relations':{},'depths':dict(collections.Counter(max(x['depth'] for x in m['gold_types']) for m in meta.values()))}
for rel in ['parents','siblings','children']:stats['relations'][rel]={'empty':0,'truncated':0}
for i,m in meta.items():
 gold={x['class_id'] for x in m['gold_types']};depth={x['class_id']:x['depth'] for x in m['gold_types']};mx=max(depth.values());gm={q for q in gold if depth[q]==mx};centre=m['reference_class_id'];mins={q for q in gold if not any(q in anc(o) for o in gold if o!=q)}
 stats['not_closed']+=bool(set.union(set(),*(set(anc(q)) for q in gold))-gold);stats['max_ties']+=len(gm)>1;stats['centre_not_max']+=centre not in gm;stats['centre_mismatch']+=centre!=sorted(mins,key=lambda q:(-depth[q],q))[0]
 m['label_in_description']=norm(m['reference_type']) in norm(m['description']);stats['description_label']+=m['label_in_description']
 rels={'parents':up[centre],'children':down[centre],'siblings':set.union(set(),*(down[q] for q in up[centre]))-{centre}}
 for rel,s in rels.items():
  stats['relations'][rel]['empty']+=not s;stats['relations'][rel]['truncated']+=len(s)>3;stats['context_mismatch']+=sorted(s)[:3]!=m['context_ids'][rel]
json.dump(stats,open('analysis/dataset_stats.json','w'),indent=2);print(stats)
rows=[]; claims=[];errors=[]
for group,records in D.items():
 for (i,c),j in records.items():
  m=meta[i]; assert j['metadata']['gold_types']==m['gold_types']
  good=j['status']=='completed';pred=j.get('predictions',[]) if good else [];pn=[norm(p) for p in pred];P=set(pn);gd={norm(x['label']):max(a['depth'] for a in m['gold_types'] if norm(a['label'])==norm(x['label'])) for x in m['gold_types']};G=set(gd);mx=max(gd.values());tp=len(P&G);p=tp/len(P) if P else 0;r=tp/len(G);f=2*tp/(len(P)+len(G));ranks=[k+1 for k,x in enumerate(pn) if gd.get(x,-1)==mx];rr=1/min(ranks) if ranks else 0
  seen=set();dcg=0
  for k,x in enumerate(pn):
   if x not in seen:dcg+=(2**(gd.get(x,0)/mx)-1)/math.log2(k+2);seen.add(x)
  ideal=sum((2**(d/mx)-1)/math.log2(k+2) for k,d in enumerate(sorted(gd.values(),reverse=True)))
  gids={x['class_id'] for x in m['gold_types']};Gup=set.union(set(gids),*(set(anc(q)) for q in gids));pup=set();cats=collections.Counter()
  for x in P:
   exact={a['class_id'] for a in m['gold_types'] if norm(a['label'])==x};qs=exact or labelmap.get(x,set());resolved=bool(exact) or len(qs)==1
   if resolved:
    pup.update(qs)
    for q in qs:pup.update(anc(q))
   else:pup.add('unresolved:'+x)
   cat='gold' if x in G else 'ancestor_outside_gold' if resolved and qs<=Gup else 'taxonomy_non_gold' if resolved else 'unresolved'
   cats[cat]+=1
   claims.append({'model':group,'entity':i,'entity_id':m['entity_id'],'name':m['entity_label'],'condition':c,'label':x,'category':cat})
  hp=len(pup&Gup)/len(pup) if pup else 0;hr=len(pup&Gup)/len(Gup);hf=2*len(pup&Gup)/(len(pup)+len(Gup))
  row=dict(model=group,entity=i,condition=c,completed=good,precision=p,recall=r,f1=f,mrr=rr,ndcg=dcg/ideal,hf1=hf,n_types=len(P),raw_n_types=len(pred),tp=tp,n_gold=len(G),depth=mx,label_in_description=m['label_in_description'],children=m['children_available'],strict=P==G,**{k:cats[k] for k in ['gold','ancestor_outside_gold','taxonomy_non_gold','unresolved']})
  u=j.get('usage',{});row['thought_tokens']=u.get('thoughtsTokenCount',np.nan);row['output_tokens']=u.get('candidatesTokenCount',j.get('generated_tokens',np.nan));rows.append(row)
  if not good:errors.append({k:j.get(k) for k in ['condition','entity_id','status','finish_reason','timestamp_utc']}|{'model':group,'entity':i})
df=pd.DataFrame(rows);df.to_csv('output/entity_scores.csv',index=False);pd.DataFrame(claims).to_csv('output/prediction_audit.csv',index=False);json.dump(errors,open('output/incomplete_results.json','w'),indent=2)
metrics=['precision','recall','f1','mrr','ndcg','hf1','n_types','thought_tokens','output_tokens']
summary=df.groupby(['model','condition'])[metrics].mean().join(df.groupby(['model','condition']).agg(n=('entity','size'),completed=('completed','sum'))).reset_index();summary.to_csv('output/condition_scores.csv',index=False)
# accepted and fixed denominators both retained
accepted=df[df.completed].groupby(['model','condition'])[metrics].mean().reset_index();accepted.to_csv('output/accepted_scores.csv',index=False)
rng=np.random.default_rng(20261013)
def contrast(a,b,kind,model,c,subset='all'):
 v=a[['entity']+metrics].merge(b[['entity']+metrics],on='entity',suffixes=('_a','_b'));d=v.f1_a-v.f1_b;n=len(d)
 if not n:return
 samples=rng.integers(0,n,(5000,n));boots=d.to_numpy()[samples].mean(axis=1);lo,hi=np.quantile(boots,[.025,.975]);out=dict(kind=kind,model=model,condition=c,subset=subset,n=n,delta_f1=d.mean(),low=lo,high=hi)
 for met in metrics:out['delta_'+met]=(v[met+'_a']-v[met+'_b']).mean()
 return out
pairs=[('graph_aware_standard','standard'),('cot_graph','cot_no_graph'),('contrastive_cot_graph','contrastive_cot_no_graph')];contr=[]
for g in D:
 sub=df[df.model==g]
 for a,b in pairs:
  for filt,ss in [('all',sub),('label present',sub[sub.label_in_description]),('label absent',sub[~sub.label_in_description]),('depth 4-5',sub[sub.depth<=5]),('depth 6-7',sub[sub.depth.between(6,7)]),('depth 8+',sub[sub.depth>=8]),('completed pairs',sub[sub.completed])]:contr.append(contrast(ss[ss.condition==a],ss[ss.condition==b],'graph',g,a,filt))
 ss=sub[~sub.children];contr.append(contrast(ss[ss.condition=='standard_children_only'],ss[ss.condition=='standard'],'empty-child control',g,'children-only'))
 contr.append(contrast(sub[sub.condition=='standard_parent_only'],sub[sub.condition=='standard_siblings_only'],'parent-sibling',g,'direct'))
for seed in ['Llama 0','Llama 42','Llama 123']:
 for c in sorted(set(df[df.model=='Gemini'].condition)):
  a=df[(df.model=='Gemini')&(df.condition==c)];b=df[(df.model==seed)&(df.condition==c)];contr.append(contrast(a,b,'model','Gemini minus '+seed,c));contr.append(contrast(a[a.completed],b[b.completed],'model complete pairs','Gemini minus '+seed,c))
for a,b in pairs:
 sub=df[df.model=='Llama 0'];sub=sub[sub.children];contr.append(contrast(sub[sub.condition==a],sub[sub.condition==a+'_no_children'],'child ablation','Llama 0',a));ss=sub[sub.completed];contr.append(contrast(ss[ss.condition==a],ss[ss.condition==a+'_no_children'],'child ablation complete pairs','Llama 0',a))
pd.DataFrame(contr).to_csv('output/paired_comparisons.csv',index=False)
# retry chronology deduplicate final copies by timestamp, input and response/status
retry=[]
for g in D:
 jobs=collections.defaultdict(dict)
 for j in list(D[g].values())+attempts[g]:
  key=(int(j['example_number']),j['condition']);fp=(j.get('timestamp_utc'),j.get('status'),j.get('response'),j.get('error'));jobs[key][str(fp)]=j
 for (i,c),js in jobs.items():
  js=sorted(js.values(),key=lambda j:j.get('timestamp_utc',''));retry.append(dict(model=g,entity=i,condition=c,attempts=len(js),retries=max(len(js)-1,0),first_valid=js[0]['status']=='completed',final_valid=D[g].get((i,c),{}).get('status')=='completed',first_status=js[0]['status'],first_date=js[0].get('timestamp_utc'),last_date=js[-1].get('timestamp_utc')))
pd.DataFrame(retry).to_csv('output/retry_audit.csv',index=False)
print(summary[['model','condition','n','completed','f1','mrr','hf1']].to_string(index=False))
