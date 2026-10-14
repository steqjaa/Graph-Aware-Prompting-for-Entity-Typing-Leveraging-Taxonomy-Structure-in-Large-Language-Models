"""Verify a machine's complete model runs and save raw outputs to a ZIP."""
from pathlib import Path
import argparse,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['gemini','llama'],required=True);a=p.parse_args()
subprocess.run([sys.executable,str(ROOT/'validate_external_inputs.py')],check=True,cwd=ROOT)
if a.model=='gemini':
 subprocess.run([sys.executable,str(ROOT/'run_gemini_external.py'),'--verify-only'],check=True,cwd=ROOT)
 folders=['results/external_demo/gemini'];name='Gemini_external_demo_results.zip'
else:
 folders=[]
 for seed in [0,42,123]:
  folder=f'results/external_demo/llama_seed_{seed}';folders.append(folder)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_llama_external.py'),'--seed',str(seed),'--max-new-tokens','0','--output-root',folder,'--verify-only'],check=True,cwd=ROOT)
 name='Llama_external_demo_three_seeds_results.zip'
with zipfile.ZipFile(ROOT/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for folder in folders+['logs','external_example','prompts_v6','scripts','archive/original/data']:
  for f in sorted((ROOT/folder).rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.name!='RUNNING.lock':z.write(f,f.relative_to(ROOT))
 for f in ['run_gemini_external.py','run_plan.json','frozen_inputs.json','README_RERUNS.md']:z.write(ROOT/f,f)
print(f'Verified {a.model} results saved: {ROOT/name}')
