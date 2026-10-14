"""Research invariants: pairing, missing-rank semantics, cohort isolation, metric edge cases."""
import csv,json,math
import pandas as pd
from common import ROOT,ORIGINAL
import score_ranked_results as s

def main():
    gold=[dict(class_id='x',label='specific',depth=2),dict(class_id='y',label='broad',depth=1)]
    assert s.reciprocal_rank(['wrong','specific'],gold,{})==.5
    assert s.reciprocal_rank(['broad'],gold,{})==0
    assert abs(s.ndcg(['specific','broad'],gold,{},2)-1)<1e-12
    assert s.ndcg(['specific','specific'],gold,{},2)<1
    ranks=pd.read_csv(ROOT/'analysis/gold_type_ranks.csv')
    assert ranks.loc[ranks.status=='missing','predicted_rank'].isna().all()
    assert ranks.loc[ranks.status=='found','predicted_rank'].ge(1).all()
    cohort=ROOT/'prompts_v4/fresh_children_100'
    manifest=pd.read_csv(cohort/'manifest.csv')
    assert manifest.entity_id.nunique()==100 and len(manifest)==900
    assert not manifest.duplicated(['entity_id','condition']).any()
    assert manifest.groupby('entity_id').condition.nunique().eq(9).all()
    oldids={json.loads(p.read_text())['entity_id'] for p in ORIGINAL.glob('generated*prompts*/example*/metadata.json')}
    assert not oldids.intersection(set(manifest.entity_id))
    for p in cohort.glob('example*/metadata.json'):
        m=json.loads(p.read_text());assert m['available_relation_counts']['children']>0
        assert all(n<=3 for n in m['selected_relation_counts'].values())
        assert m['evaluation_status']=='NOT_RUN'
    for r in manifest.itertuples():
        assert (ROOT/r.prompt_path).is_file() and (ROOT/r.metadata_path).is_file()
    checks=dict(metric_edge_cases='passed',missing_rank_semantics='passed',cohort_size=100,conditions_per_entity=9,cohort_prompt_count=900,overlap_with_attached_prior_prompt_entities=0,path_checks='passed',new_model_calls=0)
    (ROOT/'analysis/invariant_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
