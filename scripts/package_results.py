"""Package audited observed outcomes, including declared unsuccessful responses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from graph_typing.io import ROOT, validate_records

RUN_FOLDERS={'Gemini':'gemini','Llama 0':'llama_seed_0','Llama 42':'llama_seed_42','Llama 123':'llama_seed_123'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',choices=['all',*RUN_FOLDERS],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-success',action='store_true',help='Optional stricter gate; default retains failures honestly')
    a=p.parse_args(argv)
    if a.output.exists():p.error('Output exists; choose a new archive name')
    runs=list(RUN_FOLDERS) if a.run=='all' else [a.run]
    records={};files=[]
    for run in runs:
        folder=ROOT/'results/external_demo'/RUN_FOLDERS[run]
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.name in ['RUNNING.lock','.DS_Store'] or '__pycache__' in path.parts:continue
            # A result archive holds raw evidence, not large derived figures/tables.
            if 'analysis' in path.relative_to(folder).parts:continue
            files.append(path)
            if path.name.endswith('_result.json') and 'attempts' not in path.parts:
                v=json.loads(path.read_text());key=run,int(v['example_number']),v['condition']
                if key in records:raise ValueError(f'Duplicate final record {key}')
                records[key]=v
        if run!='Gemini':
            log=ROOT/f'logs/llama_seed_{run.split()[1]}.log'
            if log.exists():files.append(log)
    summary=validate_records(ROOT,records,runs=runs)
    if a.require_success and any(r['failed'] for r in summary):
        raise SystemExit('Structurally valid archive has unsuccessful responses; strict success gate rejected it')
    manifest={'scope':'audited observed results; unsuccessful records retained','runs':summary,'files':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            data=path.read_bytes();rel=str(path.relative_to(ROOT));z.writestr(rel,data)
            manifest['files'].append({'path':rel,'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('package_manifest.json',json.dumps(manifest,indent=2)+'\n')
        for rel in ['configs/final_study.json','configs/raw_archives.json']:
            z.write(ROOT/rel,rel)
    print(json.dumps(summary,indent=2));print(f'Saved {a.output.resolve()}')
    return 0

if __name__=='__main__':raise SystemExit(main())
