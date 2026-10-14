"""Compare every numeric appendix table cell with regenerated analysis.

All numeric cells, including bootstrap endpoints, are a hard gate. The
reconstructed resampling order has been verified against the saved appendix.
"""
import hashlib
import json
import re
import pandas as pd

RUNS={'Gem':'Gemini','L0':'Llama 0','L42':'Llama 42','L123':'Llama 123'}
CONDS={'S-N':'standard','S-G':'graph_aware_standard','C-N':'cot_no_graph','C-G':'cot_graph',
       'X-N':'contrastive_cot_no_graph','X-G':'contrastive_cot_graph','P':'standard_parent_only',
       'B':'standard_siblings_only','H':'standard_children_only','S-R':'graph_aware_standard_no_children',
       'C-R':'cot_graph_no_children','X-R':'contrastive_cot_graph_no_children'}

def compare(root,out):
    source=root/'manuscript/chapters/appendix_a.tex';text=source.read_text()
    def table(name):
        start=text.index(r'\renewcommand{\thetable}{'+name+'}')
        block=text[start:].split(r'\endlastfoot',1)[1].split(r'\end{longtable}',1)[0]
        return [[v.strip() for v in line.removesuffix('\\\\').split('&')] for line in block.splitlines() if '&' in line]
    score=pd.read_csv(out/'condition_summary.csv').set_index(['run','condition'])
    ext=pd.read_csv(out/'extended_condition_summary.csv').set_index(['run','condition'])
    entity=pd.read_csv(out/'entity_scores.csv');entity=entity[entity.valid]
    extended=pd.read_csv(out/'extended_entity_scores.csv')
    cats=pd.read_csv(out/'error_categories.csv').pivot(index=['run','condition'],columns='category',values='predictions').fillna(0)
    dist=pd.read_csv(out/'error_distances.csv').set_index(['run','condition'])
    pairs=pd.read_csv(out/'final_comparisons.csv')
    gain=pd.read_csv(out/'model_gain_comparisons.csv')
    person=pd.read_csv(out/'person_and_chain_comparisons.csv')
    checks=[]
    def add(table,label,metric,expected,actual,kind='point'):
        expected=float(expected);actual=float(actual)
        checks.append(dict(table=table,row=label,metric=metric,kind=kind,manuscript=expected,recomputed=actual,
                           match=abs(expected-actual)<=.00005001))
    def key(label):
        run,cond=label.split();return RUNS[run],CONDS[cond]
    for row in table('A.1'):
        actual=score.loc[key(row[0])]
        for val,col in zip(row[1:],['completed','f1','precision','recall','mrr','ndcg','raw_count']):add('A.1',row[0],col,val,actual[col])
    for row in table('A.2a'):
        actual=ext.loc[key(row[0])]
        for val,col in zip(row[1:],['micro_f1','loose_macro_f1','strict_accuracy','hierarchical_f1','hierarchy_consistency','consistency_eligible']):add('A.2a',row[0],col,val,actual[col])
    for row in table('A.2b'):
        run,cond=key(row[0]);sub=entity[(entity.run==run)&(entity.condition==cond)]
        vals=[len(sub),(sub.mrr>0).mean(),(sub.mrr==1).sum(),(sub.mrr==.5).sum(),((sub.mrr>0)&(sub.mrr<.5)).sum(),(sub.mrr==0).sum()]
        for val,col,actual in zip(row[1:],['n','hit','rank1','rank2','rank3plus','miss'],vals):add('A.2b',row[0],col,val,actual)
    for row in table('A.3a'):
        actual=cats.loc[key(row[0])]
        names=['exact_gold','broader','narrower','sibling','other_taxonomy']
        vals=[actual.sum()]+[actual.get(n,0) for n in names]+[sum(actual.get(n,0) for n in ['out_of_taxonomy','ambiguous','unconnected_label','disconnected'])]
        for val,col,value in zip(row[1:],['total',*names,'unavailable'],vals):add('A.3a',row[0],col,val,value)
    for row in table('A.3b'):
        run,cond=key(row[0]);actual=cats.loc[run,cond];dd=dist.loc[run,cond]
        sub=extended[(extended.run==run)&(extended.condition==cond)].taxonomy_error_mean
        vals=[actual.get(n,0) for n in ['out_of_taxonomy','ambiguous','unconnected_label','disconnected']]+[dd['count'],dd['mean'],sub.count(),sub.mean()]
        for val,col,value in zip(row[1:],['outside','ambiguous','no_node','disconnected','errors','pooled_mean','eligible_entities','entity_mean'],vals):add('A.3b',row[0],col,val,value)
    def style_key(label):
        run,style=label.rsplit(' ',1);return run,'Direct' if style=='Standard' else style
    def interval(table,label,cell,actual):
        nums=re.findall(r'[+-]?\d+\.\d+',cell)
        for value,col in zip(nums,['ci_low','ci_high']):add(table,label,col,value,actual[col],'bootstrap_endpoint')
    for row in table('A.4'):
        run,style=style_key(row[0]);r=pairs[(pairs.kind=='graph')&(pairs.run==run)&(pairs['style']==style)].iloc[0]
        for val,col in zip(row[1:5],['n','a_mean','b_mean','delta']):add('A.4',row[0],col,val,r[col])
        interval('A.4',row[0],row[5],r)
        for val,col in zip(row[6].split('/'),['helped','tied','harmed']):add('A.4',row[0],col,val,r[col])
    for row in table('A.5'):
        name,seed=row[0].split(' vs ');run=RUNS[seed]
        if name.endswith(' gain'):
            style=name.removesuffix(' gain');style='Direct' if style=='Standard' else style
            r=gain[(gain.run==run)&(gain['style']==style)].iloc[0]
        else:r=pairs[(pairs.kind=='model')&(pairs.run==run)&(pairs['style']==CONDS[name])].iloc[0]
        add('A.5',row[0],'n',row[1],r['n']);add('A.5',row[0],'delta',row[2],r['delta']);interval('A.5',row[0],row[3],r)
    for row in table('A.6'):
        run,style=style_key(row[0])
        for flag,offset in [(True,1),(False,4)]:
            r=person[(person.run==run)&(person['style']==style)&(person.person==flag)].iloc[0]
            label=row[0]+(' person' if flag else ' other')
            add('A.6',label,'n',row[offset],r['n'])
            numbers=re.findall(r'[+-]?\d+\.\d+',row[offset+1])
            add('A.6',label,'delta',numbers[0],r['delta'])
            interval('A.6',label,' '.join(numbers[1:]),r)
            add('A.6',label,'chain_delta',row[offset+2],r.graph_chain_rate-r.no_graph_chain_rate)
    d=pd.DataFrame(checks);d.to_csv(out/'manuscript_comparison.csv',index=False)
    failures=d[(d.kind=='point')&~d.match]
    intervals=d[(d.kind=='bootstrap_endpoint')&~d.match]
    status=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checked_values=len(d),
        point_values=int((d.kind=='point').sum()),point_mismatches=len(failures),
        bootstrap_endpoint_mismatches=len(intervals),
        explanation='Reset generator per comparison; preserve raw response order (Gemini order for model comparisons). All numeric manuscript cells are checked without changing the manuscript.')
    (out/'manuscript_checks.json').write_text(json.dumps(status,indent=2)+'\n')
    if len(failures) or len(intervals):
        raise ValueError('Manuscript numerical discrepancies: '+d[~d.match].head().to_json(orient='records'))
    print(f'Manuscript: all {len(d)} numerical values matched, including bootstrap endpoints.',flush=True)
    return status
