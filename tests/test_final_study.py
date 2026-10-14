"""Offline regression tests for final-study semantics, independent of inference."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from graph_typing.metrics import load_label_index,normalize_label,ndcg
from graph_typing.diagnostics import paired_effect
from graph_typing.io import load_records,validate_records
import pandas as pd

class FinalStudyTests(unittest.TestCase):
    def test_filtered_inventory_preserves_non_graph_ambiguity(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'labels.tsv'
            path.write_text('wd:Q1\trdfs:label\t"Camera_Model" .\nwd:Q2\trdfs:label\t"camera model" .\nwd:Q3\trdfs:label\t"other" .\n')
            all_labels,_=load_label_index(path)
            filtered,_=load_label_index(path,wanted={'camera model'})
            self.assertEqual(filtered,{'camera model':all_labels['camera model']})
            self.assertEqual(filtered['camera model'],{'wd:Q1','wd:Q2'})

    def test_pairing_uses_common_completed_entities(self):
        a=pd.Series([.1,.2,np.nan],index=[1,2,3])
        b=pd.Series([.9,.5,.3],index=[4,2,1])
        result=paired_effect(a,b)
        self.assertEqual(result['n'],2)
        self.assertAlmostEqual(result['delta'],.25)
        self.assertEqual(result['helped'],2)
        with self.assertRaises(ValueError):paired_effect(a,b.loc[[4]])

    def test_normalization_and_ranking_keep_historical_semantics(self):
        self.assertEqual(normalize_label('  CAMERA_model! '),'camera model')
        gold=[{'class_id':'a','label':'leaf','depth':2},{'class_id':'b','label':'root','depth':0}]
        self.assertEqual(ndcg(['leaf'],gold,{}),1)
        self.assertLess(ndcg(['wrong','leaf'],gold,{}),1)

    def test_raw_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs').mkdir();(root/'record.json').write_text('{}')
            item={'run':'Gemini','path':'record.json','kind':'result','sha256':'0'*64}
            (root/'configs/raw_files.jsonl').write_text(json.dumps(item)+'\n')
            with self.assertRaisesRegex(ValueError,'Raw evidence changed'):load_records(root)

    def test_unsuccessful_record_is_retained_but_missing_record_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);meta={'entity_id':'wd:Q1'};prompt='Classify'
            (root/'prompt.txt').write_text(prompt);(root/'metadata.json').write_text(json.dumps(meta))
            row={'example_number':'1','condition':'standard','entity_id':'wd:Q1','prompt_path':'prompt.txt','metadata_path':'metadata.json'}
            for part in ['all_1k','children_removed']:
                folder=root/'prompts_v6'/part;folder.mkdir(parents=True)
                with (folder/'manifest.csv').open('w') as handle:
                    w=csv.DictWriter(handle,fieldnames=list(row));w.writeheader()
                    if part=='all_1k':w.writerow(row)
            record=dict(row,metadata=meta,prompt=prompt,status='parse_error',model='gemini-3.8-flash',
                        input_hash=hashlib.sha256((prompt+json.dumps(meta,sort_keys=True)).encode()).hexdigest())
            summary=validate_records(root,{('Gemini',1,'standard'):record},runs=['Gemini'])
            self.assertEqual(summary[0]['failed'],1);self.assertEqual(summary[0]['completed'],0)
            with self.assertRaisesRegex(ValueError,'Missing scheduled record'):validate_records(root,{},runs=['Gemini'])

if __name__=='__main__':unittest.main()
