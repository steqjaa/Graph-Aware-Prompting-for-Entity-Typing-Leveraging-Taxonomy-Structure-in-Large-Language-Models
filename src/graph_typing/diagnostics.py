"""Reconstructed appendix analyses from the recorded manuscript definitions.

Core revision tables are replayed separately. Extension bootstrap RNG is reset
per comparison and source response order is preserved; this policy is recorded.
"""
import json
import numpy as np
import pandas as pd
from .metrics import Taxonomy, normalize_label, hit_at_k

PAIRS = [('standard','graph_aware_standard','Direct'), ('cot_no_graph','cot_graph','CoT'),
         ('contrastive_cot_no_graph','contrastive_cot_graph','Contrastive')]

def paired_effect(a, b):
    joined = a.rename('a').to_frame().join(b.rename('b'), how='inner', sort=False).dropna()
    if joined.empty:
        raise ValueError('No completed paired observations')
    delta = (joined.b - joined.a).to_numpy()
    rng = np.random.default_rng(230917)
    boot = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)])
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(n=len(delta), a_mean=joined.a.mean(), b_mean=joined.b.mean(), delta=delta.mean(),
                ci_low=lo, ci_high=hi, helped=int((delta>1e-12).sum()),
                tied=int((abs(delta)<=1e-12).sum()), harmed=int((delta< -1e-12).sum()))

def enrich(out, records, attempts, labels, graph):
    taxonomy = Taxonomy(graph, labels)
    rows, errors, retries, repeated_keys = [], [], [], []
    chain = ['person or organization', 'corporate body', 'object', 'entity']
    # Entity ordering keeps all runs/conditions for a gold set in the distance cache.
    for (run, example, condition), value in sorted(records.items(), key=lambda kv: (kv[0][1],kv[0][0],kv[0][2])):
        if value['status'] != 'completed':
            continue
        gold, pred = value['metadata']['gold_types'], value['predictions']
        metrics, detail = taxonomy.score(pred, gold)
        pn = [normalize_label(p) for p in pred]
        metrics.update(hit_at_1=hit_at_k(pred,gold,{},1), hit_at_3=hit_at_k(pred,gold,{},3),
                       hit_at_5=hit_at_k(pred,gold,{},5), hit_any=hit_at_k(pred,gold,{}),
                       person=any(normalize_label(g['label'])=='person' and g['depth']==4 for g in gold),
                       demo_chain_present=any(pn[i:i+4]==chain for i in range(max(0,len(pn)-3))))
        duplicate = []
        def hook(pairs):
            if len({k for k,v in pairs}) != len(pairs): duplicate.append(True)
            return dict(pairs)
        # Accepted historical parser allows markdown fences, so normalize first.
        import re
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', value['response'].strip())
        json.loads(clean, object_pairs_hook=hook)
        repeated_keys.append(dict(run=run,example=example,condition=condition,duplicate_keys=bool(duplicate)))
        rows.append(dict(run=run,example=example,condition=condition,**metrics))
        errors.extend(dict(run=run,example=example,condition=condition,**item) for item in detail)
    d, e = pd.DataFrame(rows), pd.DataFrame(errors)
    d.to_csv(out/'extended_entity_scores.csv',index=False)
    e.to_csv(out/'prediction_errors.csv',index=False)
    pd.DataFrame(repeated_keys).to_csv(out/'duplicate_key_audit.csv',index=False)
    metrics = ['f1','precision','recall','hierarchical_f1','hierarchical_precision','hierarchical_recall',
               'strict_accuracy','hierarchy_consistency','hierarchy_consistent_set','hit_at_1','hit_at_3','hit_at_5','hit_any',
               'taxonomy_error_mean','predicted_count','demo_chain_present']
    summary=d.groupby(['run','condition'])[metrics].mean()
    summary['completed']=d.groupby(['run','condition']).size()
    summary['consistency_eligible']=d.groupby(['run','condition']).hierarchy_consistency.count()
    totals=d.groupby(['run','condition'])[['tp','predicted_count','gold_count']].sum()
    summary['micro_precision']=totals.tp/totals.predicted_count
    summary['micro_recall']=totals.tp/totals.gold_count
    summary['micro_f1']=2*totals.tp/(totals.predicted_count+totals.gold_count)
    summary['loose_macro_f1']=2*summary.precision*summary.recall/(summary.precision+summary.recall)
    summary.to_csv(out/'extended_condition_summary.csv')
    categories=e.groupby(['run','condition','category']).size().rename('predictions').reset_index()
    categories.to_csv(out/'error_categories.csv',index=False)
    wrong=e[e.correct.eq(0)]
    dist=wrong.groupby(['run','condition']).distance.agg(['count','mean','median'])
    dist['incorrect_predictions']=wrong.groupby(['run','condition']).size()
    dist['unmeasurable']=dist.incorrect_predictions-dist['count']
    dist.to_csv(out/'error_distances.csv')
    core=pd.read_csv(out/'entity_scores.csv')
    core=core[core.valid].merge(d[['run','example','condition','person','demo_chain_present']],on=['run','example','condition'],validate='one_to_one')
    strata=[];uplifts=[]
    for run in ['Gemini','Llama 0','Llama 42','Llama 123']:
        for ng,gr,style in PAIRS:
            sub=core[core.run.eq(run)]
            for person in [True,False]:
                ss=sub[sub.person.eq(person)]
                aa=ss[ss.condition.eq(ng)].set_index('example');bb=ss[ss.condition.eq(gr)].set_index('example')
                effect=paired_effect(aa.f1,bb.f1)
                ix=aa.index.intersection(bb.index)
                strata.append(dict(run=run,style=style,person=person,**effect,
                    no_graph_chain_rate=aa.loc[ix,'demo_chain_present'].mean(),graph_chain_rate=bb.loc[ix,'demo_chain_present'].mean()))
    for seed in ['Llama 0','Llama 42','Llama 123']:
        for ng,gr,style in PAIRS:
            gains=[]
            for run in [seed,'Gemini']:
                ss=core[core.run.eq(run)]
                aa=ss[ss.condition.eq(ng)].set_index('example').f1
                bb=ss[ss.condition.eq(gr)].set_index('example').f1
                joined=aa.rename('ng').to_frame().join(bb.rename('gr'),how='inner',sort=False).dropna()
                gains.append(joined.gr-joined.ng)
            # Model contrasts use the Gemini source-record order.
            gains[0]=gains[0].reindex(gains[1].index)
            uplifts.append(dict(run=seed,style=style,sign='Gemini gain minus Llama gain',**paired_effect(*gains)))
    pd.DataFrame(strata).to_csv(out/'person_and_chain_comparisons.csv',index=False)
    pd.DataFrame(uplifts).to_csv(out/'model_gain_comparisons.csv',index=False)
    groups={}
    for key,value in records.items():groups[key]=[value]
    for attempt in attempts:
        key=attempt['_run'],int(attempt['example_number']),attempt['condition']
        if key not in groups:raise ValueError('Attempt outside scheduled experiment')
        groups[key].append(attempt)
    for (run,example,condition), values in groups.items():
        unique={json.dumps([v.get(k) for k in ['timestamp_utc','status','response','input_hash']],sort_keys=True):v for v in values}
        seq=sorted(unique.values(),key=lambda v:v.get('timestamp_utc',''))
        retries.append(dict(run=run,example=example,condition=condition,observed_attempts=len(seq),
                            observed_retries=len(seq)-1,first_valid=seq[0]['status']=='completed',
                            final_valid=records[run,example,condition]['status']=='completed'))
    rd=pd.DataFrame(retries);rd.to_csv(out/'retry_chronology.csv',index=False)
    rd.groupby(['run','condition']).agg(scheduled=('example','size'),first_valid=('first_valid','sum'),
        final_valid=('final_valid','sum'),extra_attempts=('observed_retries','sum'),
        maximum_observed_attempts=('observed_attempts','max')).to_csv(out/'retry_summary.csv')
    token_rows=[]
    for (run,example,condition), value in records.items():
        if value['status']!='completed':continue
        usage=value.get('usage',value.get('raw_response',{}).get('usageMetadata',{}))
        token_rows.append(dict(run=run,condition=condition,example=example,
            thought_tokens=usage.get('thoughtsTokenCount',np.nan),
            output_tokens=usage.get('candidatesTokenCount',value.get('generated_tokens',np.nan))))
    pd.DataFrame(token_rows).groupby(['run','condition'])[['thought_tokens','output_tokens']].agg(['count','mean','median','sum']).to_csv(out/'token_summary.csv')
    (out/'extension_provenance.json').write_text(json.dumps(dict(
        implementation='reconstructed from manuscript definitions; original unsaved extension scripts unavailable',
        bootstrap_resamples=2000,bootstrap_seed=230917,rng='reset per named comparison',
        row_order='source response order; Gemini source order for model contrasts',cohort='intersection of all required completed outputs',
        error_precedence=['exact_gold','broader','narrower','sibling','other'],
        person='gold contains normalized person at depth 4',
        chain='consecutive normalized person or organization, corporate body, object, entity',
        retry_scope='observed records only; no inference of unrecorded attempts'),indent=2)+'\n')
