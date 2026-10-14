"""Exercise optional plotting paths using explicitly synthetic temporary comparisons."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class PlotIntegrationTests(unittest.TestCase):
    def test_all_optional_figures_and_mismatched_child_rejection(self):
        source = ROOT / 'results/gemini_1k_fresh/analysis'
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            primary, child = temp/'primary', temp/'children'
            (primary/'analysis').mkdir(parents=True)
            (child/'analysis').mkdir(parents=True)
            for name in ['entity_scores.csv', 'completion_status.csv', 'interpretation.json', 'prediction_errors.csv']:
                shutil.copyfile(source/name, primary/'analysis'/name)
            scores = pd.read_csv(source/'entity_scores.csv')
            graphs = ['graph_aware_standard','cot_graph','contrastive_cot_graph']
            sub = scores[scores.children_available & scores.condition.isin(graphs)].copy()
            sub['condition'] += '_no_children'
            sub['source_prompt_sha256'] = sub.prompt_sha256
            sub['model_version'] = 'TEST_FIXTURE_NOT_INFERENCE'
            sub.to_csv(child/'analysis/entity_scores.csv',index=False)
            sub[['entity_id','condition']].assign(status='completed').to_csv(child/'analysis/completion_status.csv',index=False)
            (child/'analysis/interpretation.json').write_text(json.dumps({'schema_version':2}))
            cmd = [sys.executable, str(ROOT/'plot_stories.py'), '--results', str(primary),
                   '--primary-label', 'TEST PRIMARY', '--compare-results', str(primary),
                   '--compare-label', 'TEST IDENTICAL COMPARISON', '--children-results', str(child)]
            result = subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            figures = primary/'analysis/story_figures'
            self.assertEqual(len(list(figures.glob('*.png'))),15)
            model = pd.read_csv(figures/'10_model_paired_effects.csv')
            matched = pd.read_csv(figures/'12_matched_children.csv')
            self.assertTrue((model.mean_delta == 0).all())
            self.assertTrue((matched.mean_delta == 0).all())
            self.assertTrue((matched.n == 451).all())
            # A missing entity must not turn into a silent inner-join comparison.
            sub = sub[sub.entity_id != sub.entity_id.iloc[0]]
            sub.to_csv(child/'analysis/entity_scores.csv',index=False)
            sub[['entity_id','condition']].assign(status='completed').to_csv(child/'analysis/completion_status.csv',index=False)
            result = subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('cohorts differ',result.stderr)

if __name__ == '__main__': unittest.main()
