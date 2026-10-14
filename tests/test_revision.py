"""Run with: python -m unittest discover -s tests -v (from project root)."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from evaluation import Taxonomy, paired_summary, set_metrics
from prepare_children_ablation import remove_children
from run_llama import parse, digest, MODEL, REVISION
from score_gemini import summarize
from plot_stories import aligned

class MetricsTests(unittest.TestCase):
    def setUp(self):
        g = nx.DiGraph([('leaf','parent'),('parent','root'),('sibling','parent'),
                        ('narrow','leaf'),('other','root'),('isolated','separate')])
        labels = {v: {v} for v in g}
        labels['ambiguous'] = {'leaf', 'other'}
        self.tax = Taxonomy(g, labels)
        self.gold = [dict(label='leaf', class_id='leaf', depth=2)]

    def test_flat_duplicates_and_extra_labels(self):
        m, _ = self.tax.score(['leaf', 'leaf', 'unknown'], self.gold)
        self.assertEqual(m['predicted_count'], 2)
        self.assertAlmostEqual(m['f1'], 2/3)
        self.assertAlmostEqual(m['hierarchical_f1'], 6/7)
        self.assertEqual(m['strict_accuracy'], 0)

    def test_ancestor_partial_credit(self):
        near, _ = self.tax.score(['parent'], self.gold)
        far, _ = self.tax.score(['other'], self.gold)
        self.assertEqual(near['f1'], 0)
        self.assertAlmostEqual(near['hierarchical_f1'], 4/5)
        self.assertGreater(near['hierarchical_f1'], far['hierarchical_f1'])

    def test_dag_multiple_parents(self):
        g = nx.DiGraph([('leaf','a'),('leaf','b'),('a','root'),('b','root')])
        t = Taxonomy(g, {v: {v} for v in g})
        metrics, _ = t.score(['leaf','a','root'], self.gold)
        self.assertAlmostEqual(metrics['hierarchy_consistency'], 2/3)
        self.assertEqual(metrics['hierarchy_consistent_set'], 0)
        metrics, _ = t.score(['leaf','a','b','root'], self.gold)
        self.assertEqual(metrics['hierarchy_consistent_set'], 1)

    def test_distance_direction_ambiguity_disconnected(self):
        _, rows = self.tax.score(['parent','root','narrow','sibling','other','ambiguous','isolated','xyz'], self.gold)
        r = {x['prediction']:x for x in rows}
        self.assertEqual(r['parent']['error_bucket'], 'Broader 1')
        self.assertEqual(r['root']['direction_distance'], 2)
        self.assertEqual(r['narrow']['category'], 'narrower')
        self.assertEqual(r['sibling']['category'], 'sibling')
        self.assertEqual(r['sibling']['distance'], 2)
        self.assertEqual(r['other']['distance'], 3)
        self.assertEqual(r['ambiguous']['category'], 'ambiguous')
        self.assertIsNone(r['ambiguous']['distance'])
        self.assertEqual(r['isolated']['category'], 'disconnected')
        self.assertEqual(r['xyz']['category'], 'out_of_taxonomy')

    def test_no_predictions_and_no_resolved_nodes(self):
        import math
        m, _ = self.tax.score([], self.gold)
        self.assertEqual(m['f1'], 0)
        self.assertTrue(math.isnan(m['hierarchy_consistency']))
        self.assertEqual(set_metrics([], [])['f1'], 0)

    def test_micro_differs_from_entity_macro(self):
        from score_gemini import METRICS
        data = []
        for pred, gold in [(['a'], ['a']), ([], ['a','b','c'])]:
            row = dict.fromkeys(METRICS, 0.0)
            row.update(set_metrics(pred, gold)); row['condition'] = 'standard'
            data.append(row)
        s = summarize(pd.DataFrame(data)).loc['standard']
        self.assertEqual(s.macro_f1, .5)
        self.assertEqual(s.micro_f1, .4)

    def test_bootstrap_and_alignment(self):
        s = paired_summary([.1,.1,.1])
        self.assertAlmostEqual(s['ci_low'], .1)
        self.assertEqual(s['helped'], 3)
        self.assertEqual(paired_summary([0,0])['tied'], 2)
        d = pd.DataFrame(dict(entity_id=['a','b'], gold_signature=['g','h'], f1=[.1,.2], input_hash=['x','y']))
        self.assertTrue((aligned(d, d.iloc[::-1], 'f1', True) == 0).all())
        with self.assertRaises(ValueError): aligned(d, d.iloc[:1], 'f1')
        bad = d.copy(); bad.loc[0, 'gold_signature'] = 'bad'
        with self.assertRaises(ValueError): aligned(d, bad, 'f1')

class ExperimentTests(unittest.TestCase):
    def test_response_validation(self):
        self.assertEqual(parse('{"types":["person"]}', 'standard')[0], 'completed')
        self.assertEqual(parse('```json\n{"types":["person"]}\n```', 'standard')[0], 'completed')
        self.assertEqual(parse('{"types":["person"]}', 'cot_graph')[0], 'parse_error')
        self.assertEqual(parse('{"types":[]}', 'standard')[0], 'parse_error')
        self.assertEqual(parse('{"types":["person"]}', 'standard', False)[0], 'incomplete')

    def test_full_child_cohort_and_preserved_demonstrations(self):
        with (ROOT / 'prompts_v5/all_1k/manifest.csv').open() as f:
            main = list(csv.DictReader(f))
        with (ROOT / 'prompts_v5/children_removed/manifest.csv').open() as f:
            child = list(csv.DictReader(f))
        self.assertEqual(len(main), 9000)
        self.assertEqual(len(child), 1353)
        self.assertEqual(len({r['entity_id'] for r in child}), 451)
        lookup = {(r['entity_id'], r['condition']):r for r in main}
        for row in child:
            source = lookup[row['entity_id'], row['source_condition']]
            before = (ROOT / source['prompt_path']).read_text()
            after = (ROOT / row['prompt_path']).read_text()
            self.assertEqual(after, remove_children(before))
            self.assertEqual(digest(before), row['source_prompt_sha256'])
            self.assertNotIn('- Direct child classes:', after.split('Now classify:')[1])
            if 'Example:' in before:
                self.assertEqual(before.split('Example:')[1].split('Now classify:')[0],
                                 after.split('Example:')[1].split('Now classify:')[0])
            for tag in ['- Direct parent classes:', '- Sibling classes:']:
                self.assertEqual([l for l in before.splitlines() if l.startswith(tag)],
                                 [l for l in after.splitlines() if l.startswith(tag)])

    def test_runner_verify_and_input_change_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompt, meta = tmp/'prompt.txt', tmp/'metadata.json'
            prompt.write_text('Classify test'); meta.write_text('{"entity_id":"test"}')
            row = dict(example_number=1, entity_id='test', entity_label='Test', condition='standard',
                       prompt_path=str(prompt), metadata_path=str(meta))
            mf = tmp/'manifest.csv'
            with mf.open('w', newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            out=tmp/'results';out.mkdir()
            fp=digest(prompt.read_text()+json.dumps(json.loads(meta.read_text()),sort_keys=True))
            runner=ROOT/'scripts/run_llama.py'
            cfg=dict(model=MODEL,revision=REVISION,dtype='bfloat16',do_sample=False,max_new_tokens=800,batch_size=8,
                     manifest_sha256=digest(mf.read_text()),inputs_sha256=digest(fp),runner_sha256=digest(runner.read_text()))
            (out/'run_config.json').write_text(json.dumps(cfg))
            target=out/'example_0001/standard_result.json';target.parent.mkdir()
            x=dict(row,input_hash=fp,model=MODEL,revision=REVISION,status='completed',response='{"types":["person"]}',
                   predictions=['person'],explanation='',finish_reason='STOP')
            target.write_text(json.dumps(x))
            cmd=[sys.executable,str(runner),'--manifest',str(mf),'--output',str(out),'--verify-only']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            x['predictions']=['wrong'];target.write_text(json.dumps(x))
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            prompt.write_text('changed')
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('changed', result.stderr)

if __name__ == '__main__': unittest.main()
