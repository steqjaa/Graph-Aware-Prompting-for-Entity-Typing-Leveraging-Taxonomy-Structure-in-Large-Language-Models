"""Final manuscript bootstrap: reset per comparison, preserve source entity order."""
import pandas as pd
from .diagnostics import paired_effect,PAIRS

def replay(out):
    d=pd.read_csv(out/'entity_scores.csv');d=d[d.valid]
    entities=pd.read_csv(out/'entities.csv').set_index('example')
    rows=[]
    def series(run,condition,ids=None):
        s=d[(d.run==run)&(d.condition==condition)].set_index('example').f1
        return s if ids is None else s[s.index.isin(ids)]
    for run in ['Gemini','Llama 0','Llama 42','Llama 123']:
        for ng,gr,style in PAIRS:
            for kind,stratum,ids in [('graph','',None),
                ('label_in_description','False',entities[~entities.label_in_description].index),
                ('label_in_description','True',entities[entities.label_in_description].index),
                ('depth','4–5',entities[entities.depth.between(4,5)].index),
                ('depth','6–7',entities[entities.depth.between(6,7)].index),
                ('depth','8+',entities[entities.depth>=8].index)]:
                rows.append(dict(kind=kind,run=run,style=style,stratum=stratum,
                    **paired_effect(series(run,ng,ids),series(run,gr,ids))))
        ids=entities[~entities.children_available].index
        rows.append(dict(kind='empty_children',run=run,style='Direct',stratum='',
            **paired_effect(series(run,'standard',ids),series(run,'standard_children_only',ids))))
    for seed in ['Llama 0','Llama 42','Llama 123']:
        for condition in [c for pair in PAIRS for c in pair[:2]]+['standard_parent_only','standard_siblings_only','standard_children_only']:
            gem=series('Gemini',condition);llama=series(seed,condition).reindex(gem.index)
            rows.append(dict(kind='model',run=seed,style=condition,stratum='',**paired_effect(llama,gem)))
    for _,gr,style in PAIRS:
        rows.append(dict(kind='child',run='Llama 0',style=style,stratum='',
            **paired_effect(series('Llama 0',gr+'_no_children'),series('Llama 0',gr))))
    result=pd.DataFrame(rows)
    result.to_csv(out/'final_comparisons.csv',index=False)
    return result
