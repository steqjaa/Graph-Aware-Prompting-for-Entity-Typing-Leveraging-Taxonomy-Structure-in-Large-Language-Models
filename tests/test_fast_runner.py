"""GPU-free tests of real resume/retry orchestration with an injected engine."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_llama_fast as fast


class Tokenizer:
    eos_token_id = 0
    def convert_tokens_to_ids(self, text):
        return 1
    def apply_chat_template(self, messages, **kwargs):
        # New Transformers versions return a dict by default.
        return [2, 3, 4] if kwargs.get('return_dict') is False else {'input_ids': [2, 3, 4]}


class Engine:
    loads = 0
    def __init__(self, **kwargs):
        Engine.loads += 1
        self.calls = 0
    def get_tokenizer(self):
        return Tokenizer()
    def generate(self, prompts, params, **kwargs):
        self.calls += 1
        outputs = []
        for prompt in prompts:
            assert isinstance(prompt['prompt_token_ids'], list)
            assert all(isinstance(token, int) for token in prompt['prompt_token_ids'])
            answer = SimpleNamespace(text='{"types":["person"],"explanation":"Reason"}',
                finish_reason='length' if self.calls == 1 else 'stop', token_ids=[4, 5])
            outputs.append(SimpleNamespace(outputs=[answer], prompt_token_ids=prompt['prompt_token_ids']))
        return outputs


class FastRunnerTests(unittest.TestCase):
    def test_budget_no_ceiling_and_context_bound(self):
        self.assertEqual(fast.budget(1000, 32768, 0), 31768)
        self.assertEqual(fast.budget(1000, 32768, 16384), 16384)
        self.assertEqual(fast.budget(30000, 32768, 16384), 2768)
        with self.assertRaises(ValueError):
            fast.budget(32768, 32768, 0)

    def test_structured_limits_and_required_fields(self):
        self.assertTrue(fast.schema_valid('{"types":["person"]}', 'standard'))
        self.assertFalse(fast.schema_valid('{"types":["person"]}', 'cot_graph'))
        self.assertTrue(fast.schema_valid('{"explanation":"Reason","types":["person"]}', 'cot_graph'))
        self.assertFalse(fast.schema_valid(json.dumps({'explanation':'a'*1201,'types':['person']}), 'cot_graph'))
        self.assertFalse(fast.schema_valid(json.dumps({'types':['person']*33}), 'standard'))
        self.assertFalse(fast.schema_valid('{"types":["person"],"extra":1}', 'standard'))
        self.assertEqual(list(fast.response_schema('contrastive_cot_no_graph')['properties']), ['explanation','types'])

    def test_structured_factory_is_passed_to_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'scripts').mkdir()
            shutil.copyfile(ROOT/'scripts/run_llama.py',root/'scripts/run_llama.py')
            (root/'prompt.txt').write_text('Classify')
            (root/'meta.json').write_text('{"entity_id":"one"}')
            row=dict(example_number=1,entity_id='one',entity_label='One',condition='cot_graph',
                     prompt_path='prompt.txt',metadata_path='meta.json')
            with (root/'manifest.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            class StructuredEngine(Engine):
                def __init__(self, **kwargs):
                    assert kwargs['structured_outputs_config']['backend']=='xgrammar'
                    super().__init__(**kwargs)
                def generate(self, prompts, params, **kwargs):
                    assert params[0].structured_outputs.json['properties']['explanation']['maxLength']==1200
                    return super().generate(prompts,params,**kwargs)
            with patch.object(fast,'ROOT',root),patch.object(fast,'EXPERIMENTS',{'main':('manifest.csv','main')}):
                self.assertEqual(fast.main(['--experiment','main','--structured-json'],
                    StructuredEngine, SimpleNamespace, SimpleNamespace),0)

    def test_retry_archive_resume_integrity_and_no_gpu_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'scripts').mkdir()
            shutil.copyfile(ROOT/'scripts/run_llama.py', root/'scripts/run_llama.py')
            (root/'prompt.txt').write_text('Classify a person')
            (root/'meta.json').write_text('{"entity_id":"one"}')
            row = dict(example_number=1,entity_id='one',entity_label='One',condition='cot_graph',
                       prompt_path='prompt.txt',metadata_path='meta.json')
            with (root/'manifest.csv').open('w', newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            with patch.object(fast, 'ROOT', root), patch.object(fast, 'EXPERIMENTS', {'main':('manifest.csv','main')}):
                Engine.loads=0
                self.assertEqual(fast.main(['--experiment','main','--dry-run']), 0)
                self.assertFalse((root/'results').exists())
                self.assertEqual(fast.main(['--experiment','main'], Engine, SimpleNamespace), 0)
                self.assertEqual(Engine.loads,1)
                saved=json.loads((root/'results/main/example_0001/cot_graph_result.json').read_text())
                self.assertEqual(saved['finish_reason'],'STOP')
                self.assertEqual(saved['status'],'completed')
                self.assertEqual(saved['generation_config']['max_tokens'],32765)
                archived=list((root/'results/main/attempts').glob('*.json'))
                self.assertEqual(len(archived),1)
                self.assertEqual(json.loads(archived[0].read_text())['status'],'incomplete')
                self.assertEqual(fast.main(['--experiment','main','--verify-only']),0)
                self.assertEqual(fast.main(['--experiment','main'], Engine, SimpleNamespace),0)
                self.assertEqual(Engine.loads,1)
                saved['predictions']=['tampered']
                (root/'results/main/example_0001/cot_graph_result.json').write_text(json.dumps(saved))
                self.assertEqual(fast.main(['--experiment','main','--verify-only']),2)
                (root/'prompt.txt').write_text('Changed prompt')
                with self.assertRaises(SystemExit):
                    fast.main(['--experiment','main','--verify-only'])


if __name__ == '__main__':
    unittest.main()
