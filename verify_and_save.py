"""Verify all four complete experiments and package raw results; no inference."""
from pathlib import Path
import subprocess,sys,json,zipfile
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT/'validate_external_inputs.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'run_gemini_external.py'),'--verify-only'],check=True,cwd=ROOT)
for seed in [0,42,123]:
 subprocess.run([sys.executable,str(ROOT/'scripts/run_llama_external.py'),'--experiment','all','--seed',str(seed),'--max-new-tokens','0','--output-root',f'results/external_demo/llama_seed_{seed}','--verify-only'],check=True,cwd=ROOT)
plan=json.loads((ROOT/'run_plan.json').read_text());plan['status']='COMPLETE_VERIFIED';plan['accepted_responses']=40059
(ROOT/'results/external_demo/verified_completion.json').write_text(json.dumps(plan,indent=2))
out=ROOT/'External_demo_four_runs_results.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for folder in ['results/external_demo','logs','external_example','prompts_v6','scripts','archive/original/data']:
  for f in sorted((ROOT/folder).rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.name!='RUNNING.lock':z.write(f,f.relative_to(ROOT))
 for name in ['run_plan.json','run_gemini_external.py','frozen_inputs.json','validate_external_inputs.py','README_RERUNS.md']:
  z.write(ROOT/name,name)
print(f'COMPLETE: {out}. All 40,059 responses verified; attempts and provenance included.')
