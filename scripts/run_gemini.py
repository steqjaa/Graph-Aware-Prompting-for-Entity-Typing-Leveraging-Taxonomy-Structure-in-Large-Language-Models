"""Resumable Gemini generateContent REST runner. One independent request per prompt.
API: https://ai.google.dev/api/generate-content ; https://ai.google.dev/api/models
No API key is saved. Raw responses and each attempt are retained.
"""
import argparse,csv,json,os,time,hashlib,threading,urllib.request,urllib.error,concurrent.futures,random,getpass
from datetime import datetime,timezone
from common import ROOT
BASE='https://generativelanguage.googleapis.com/v1beta/'
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)
def parse_response(raw):
 cs=raw.get('candidates',[])
 if not cs:return 'blocked','',[],'',None
 c=cs[0];text=''.join(x.get('text','') for x in c.get('content',{}).get('parts',[]) if not x.get('thought'))
 finish=c.get('finishReason')
 if finish!='STOP':return 'incomplete',text,[],'',finish
 try:
  obj=json.loads(text);pred=obj['types']
  if not isinstance(pred,list) or any(not isinstance(x,str) or not x.strip() for x in pred):raise ValueError()
  exp=obj.get('explanation','')
  if not isinstance(exp,str):raise ValueError()
  return 'completed',text,pred,exp,finish
 except (ValueError,KeyError,TypeError):return 'parse_error',text,[],'',finish

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=str,default='prompts_v5/all_1k/manifest.csv');p.add_argument('--output',default='results/gemini_1k');p.add_argument('--model',default='gemini-3.8-flash');p.add_argument('--list-models',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-calls',type=int);p.add_argument('--workers',type=int,default=4);p.add_argument('--rpm',type=float,default=30);p.add_argument('--retry-failed',action='store_true');p.add_argument('--max-output-tokens',type=int,default=4096);p.add_argument('--temperature',type=float,default=1.0);a=p.parse_args()
 if a.workers<1 or a.rpm<=0 or (a.max_calls is not None and a.max_calls<1):p.error('workers, rpm, max-calls must be positive')
 key=None
 def request(path,body=None):
  data=None if body is None else json.dumps(body).encode()
  req=urllib.request.Request(BASE+path,data=data,headers={'x-goog-api-key':key,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
 if a.list_models:
  key=os.environ.get('GEMINI_API_KEY') or getpass.getpass('Gemini API key (hidden): ')
  page=''
  while True:
   r=request('models?pageSize=1000'+('&pageToken='+urllib.parse.quote(page) if page else ''))
   for m in r.get('models',[]):
    if 'generateContent' in m.get('supportedGenerationMethods',[]):print(m['name'].removeprefix('models/'))
   page=r.get('nextPageToken')
   if not page:break
  return
 mf=ROOT/a.manifest;rows=list(csv.DictReader(mf.open()));seen=set();fingerprints=[]
 for row in rows:
  k=(row['entity_id'],row['condition'])
  if k in seen:raise SystemExit('Duplicate entity/condition in manifest')
  seen.add(k)
  text=(ROOT/row['prompt_path']).read_text();meta=json.loads((ROOT/row['metadata_path']).read_text())
  if meta['entity_id']!=row['entity_id']:raise SystemExit('Metadata entity mismatch')
  fingerprints.append(digest(text+json.dumps(meta,sort_keys=True)))
 config=dict(model=a.model.removeprefix('models/'),temperature=a.temperature,max_output_tokens=a.max_output_tokens,thinking='provider default; no explicit override',manifest_sha256=digest(mf.read_text()),inputs_sha256=digest(''.join(fingerprints)),response_format='JSON MIME, prompt-defined schema',runner_sha256=digest(open(__file__).read()))
 out=ROOT/a.output;cfg=out/'run_config.json'
 if cfg.exists() and json.loads(cfg.read_text())!=config:raise SystemExit('Run configuration/input changed. Use a new --output; do not mix experiments.')
 pending=[];skipped=0;failed=0
 for row,fp in zip(rows,fingerprints):
  target=out/f"example_{int(row['example_number']):04d}"/f"{row['condition']}_result.json"
  if target.exists():
   existing=json.loads(target.read_text())
   if existing.get('input_hash')!=fp:raise SystemExit('Input changed for existing result')
   if existing.get('status')=='completed':skipped+=1;continue
   if not a.retry_failed:failed+=1;continue
  pending.append((row,fp,target))
 print(f'Entities: {len({r["entity_id"] for r in rows})}; conditions: {len({r["condition"] for r in rows})}; total: {len(rows)}; completed: {skipped}; saved failures not retried: {failed}; pending: {len(pending)}')
 if a.max_calls:pending=pending[:a.max_calls]
 print(f'This invocation: up to {len(pending)} prompt jobs, plus transient retries. Model: {config["model"]}')
 if a.dry_run:return
 key=os.environ.get('GEMINI_API_KEY') or getpass.getpass('Gemini API key (hidden): ')
 if not key:raise SystemExit('No API key supplied')
 # Validate model access before queuing thousands of requests.
 info=request('models/'+urllib.parse.quote(config['model'],safe=''))
 if 'generateContent' not in info.get('supportedGenerationMethods',[]):raise SystemExit('Model does not advertise generateContent')
 out.mkdir(parents=True,exist_ok=True)
 lockfile=out/'RUNNING.lock'
 try:fd=os.open(lockfile,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:raise SystemExit('Run already locked. If no process is running, remove RUNNING.lock then resume.')
 os.write(fd,str(os.getpid()).encode());os.close(fd)
 save(cfg,config);save(out/'model_info.json',info)
 gate=threading.Lock();next_start=[0.];stop=threading.Event()
 def pace():
  with gate:
   now=time.monotonic();delay=max(0,next_start[0]-now);next_start[0]=max(now,next_start[0])+60/a.rpm
  if delay:time.sleep(delay)
 def run(job):
  row,fp,target=job
  if stop.is_set():return
  text=(ROOT/row['prompt_path']).read_text();meta=json.loads((ROOT/row['metadata_path']).read_text())
  record=dict(row,example_number=int(row['example_number']),input_hash=fp,model=config['model'],prompt=text,metadata=meta,metadata_path=row['metadata_path'],status='api_error',predictions=[])
  for attempt in range(1,5):
   if stop.is_set():return
   pace();record['timestamp_utc']=datetime.now(timezone.utc).isoformat()
   try:
    raw=request('models/'+urllib.parse.quote(config['model'],safe='')+':generateContent',{'contents':[{'role':'user','parts':[{'text':text}]}],'generationConfig':{'temperature':a.temperature,'maxOutputTokens':a.max_output_tokens,'responseMimeType':'application/json'}})
    status,response,pred,exp,finish=parse_response(raw)
    if 'cot' in row['condition'] and status=='completed' and not exp.strip():status='parse_error'
    record.update(status=status,response=response,predictions=pred,explanation=exp,finish_reason=finish,raw_response=raw,model_version=raw.get('modelVersion'),usage=raw.get('usageMetadata',{}))
    retry=False
   except urllib.error.HTTPError as ex:
    code=ex.code;record.update(status='api_error',http_status=code,error=f'HTTP {code}')
    retry=code in (429,500,502,503,504)
    if code in (400,401,403,404):stop.set()
   except (urllib.error.URLError,TimeoutError) as ex:
    record.update(status='api_error',error=type(ex).__name__);retry=True
   attempt_path=out/'attempts'/f"{row['example_number']}_{row['condition']}_{time.time_ns()}.json"
   save(attempt_path,record)
   if not retry or attempt==4:break
   time.sleep(min(60,2**attempt+random.random()))
  save(target,record)
  if record.get('http_status')==429 and record['status']=='api_error':stop.set()
  print(f"{row['example_number']} {row['condition']}: {record['status']}",flush=True)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
   for _ in executor.map(run,pending):pass
 finally:lockfile.unlink(missing_ok=True)
 print('Finished. Rerun to resume; use --retry-failed to retry saved failures. Score command reports completeness.')
if __name__=='__main__':main()
