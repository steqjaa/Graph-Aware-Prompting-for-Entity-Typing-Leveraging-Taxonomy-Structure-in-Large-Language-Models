"""GPU-free orchestration checks. All simulated outputs stay in temporary folders."""
import unittest,tempfile,json,csv,shutil,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_llama_external as runner
class Tokenizer:
 eos_token_id=0
 def convert_tokens_to_ids(self,text):return 1
 def apply_chat_template(self,messages,**kwargs):return [2,3,4]
class Engine:
 expected_seed=0
 loads=0
 def __init__(self,**kwargs):
  assert kwargs['seed']==self.expected_seed
  assert kwargs['revision']==runner.REVISION
  Engine.loads+=1;self.calls=0
 def get_tokenizer(self):return Tokenizer()
 def generate(self,prompts,params,**kwargs):
  self.calls+=1
  for p in params:
   assert p.seed==self.expected_seed and p.temperature==1.0 and p.top_p==1
   assert p.structured_outputs.json_object is True
  return [SimpleNamespace(outputs=[SimpleNamespace(text='{"explanation":"Reason","types":["person"]}',finish_reason='length' if self.calls==1 else 'stop',token_ids=[4,5])],prompt_token_ids=[2,3,4]) for x in prompts]
class Tests(unittest.TestCase):
 def test_seeds_retries_resume_and_integrity(self):
  for seed in [0,42,123]:
   with self.subTest(seed=seed),tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'scripts').mkdir();shutil.copy2(ROOT/'scripts/run_llama.py',root/'scripts/run_llama.py')
    (root/'prompt.txt').write_text('Classify');(root/'meta.json').write_text('{"entity_id":"one"}')
    row=dict(example_number=1,entity_id='one',entity_label='One',condition='cot_graph',prompt_path='prompt.txt',metadata_path='meta.json')
    with (root/'manifest.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    args=['--experiment','main','--output-root',f'results/seed_{seed}','--seed',str(seed),'--max-new-tokens','0']
    Engine.expected_seed=seed;Engine.loads=0
    with patch.object(runner,'ROOT',root),patch.object(runner,'EXPERIMENTS',{'main':('manifest.csv','main')}):
     self.assertEqual(runner.main(args+['--dry-run']),0);self.assertFalse((root/'results').exists())
     self.assertEqual(runner.main(args,Engine,SimpleNamespace,SimpleNamespace),0)
     out=root/f'results/seed_{seed}/main';result=out/'example_0001/cot_graph_result.json'
     saved=json.loads(result.read_text());self.assertEqual(saved['generation_config']['seed'],seed)
     self.assertEqual(json.loads((out/'run_config.json').read_text())['seed'],seed)
     attempts=list((out/'attempts').glob('*.json'));self.assertEqual(len(attempts),1)
     self.assertEqual(json.loads(attempts[0].read_text())['generation_config']['seed'],seed)
     self.assertEqual(runner.main(args+['--verify-only']),0)
     self.assertEqual(runner.main(args,Engine,SimpleNamespace,SimpleNamespace),0);self.assertEqual(Engine.loads,1)
     other=args.copy();other[other.index('--seed')+1]=str(seed+1)
     with self.assertRaises(SystemExit):runner.main(other+['--verify-only'])
     saved['generation_config']['seed']=seed+1;result.write_text(json.dumps(saved))
     with self.assertRaises(ValueError):runner.main(args+['--verify-only'])
     saved['generation_config']['seed']=seed;result.write_text(json.dumps(saved))
     (root/'prompt.txt').write_text('Changed input')
     with self.assertRaises(SystemExit):runner.main(args+['--verify-only'])
if __name__=='__main__':unittest.main()
