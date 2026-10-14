"""Entity inventory and observed gold-taxonomy categories; no invented domains."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/final_comparison/entity_distribution'
    out.mkdir(parents=True,exist_ok=True)
    rows=[];members=defaultdict(list)
    for p in sorted((ROOT/'prompts_v5/all_1k').glob('example_*/metadata.json')):
        x=json.loads(p.read_text())
        categories=sorted({g['label'] for g in x['gold_types'] if g['depth']==4})
        if not categories:categories=['No gold label at depth 4']
        row=dict(entity_id=x['entity_id'],entity=x['entity_label'],description=x['description'],
            reference_type=x['reference_type'],broad_gold_categories=' | '.join(categories),
            maximum_gold_depth=max(g['depth'] for g in x['gold_types']),
            children_available=x['children_available'],
            gold_types=' | '.join(g['label'] for g in x['gold_types']))
        rows.append(row)
        for c in categories:members[c].append(row)
    df=pd.DataFrame(rows);assert len(df)==1000 and df.entity_id.nunique()==1000
    df.to_csv(out/'all_1000_entities.csv',index=False)
    cats=pd.DataFrame([dict(category=c,entity_count=len(v),share_of_1000=len(v)/1000,
        children_present=sum(bool(r['children_available']) for r in v),
        example_entities=' | '.join(r['entity'] for r in v[:3])) for c,v in members.items()])
    cats=cats.sort_values(['entity_count','category'],ascending=[False,True])
    cats.to_csv(out/'broad_category_distribution.csv',index=False)
    niches=df.groupby('reference_type').agg(entity_count=('entity_id','size'),
        example_entities=('entity',lambda s:' | '.join(s.head(3)))).reset_index()
    niches['share_of_1000']=niches.entity_count/1000
    niches=niches.sort_values(['entity_count','reference_type'],ascending=[False,True])
    niches.to_csv(out/'specific_type_distribution.csv',index=False)
    df.groupby('maximum_gold_depth').size().rename('entities').to_csv(out/'target_depth_distribution.csv')
    lines=['# Entity composition','',
        'This table describes the 1,000 unique entities, not 9,000 repeated prompt jobs. '
        'Broad categories are observed gold labels at taxonomy depth 4. This is a reproducible descriptive level, '
        'not an independently annotated domain classification. Entities may belong to multiple categories; '
        'category counts therefore overlap and shares need not sum to 100%. '
        'Niches use each entity’s recorded reference_type, one per entity.', '',
        f'There are {len(niches)} distinct reference types; '
        f'{int((niches.entity_count==1).sum())} occur only once. '
        f'{int(df.children_available.sum())} entities have child context.', '',
        '| Broad gold category | Entities | Share of 1,000 | Examples |',
        '|---|---:|---:|---|']
    for r in cats.itertuples():
        examples=r.example_entities.replace('|',';')
        lines.append(f'| {r.category} | {r.entity_count} | {r.share_of_1000:.1%} | {examples} |')
    lines+=['','## Most frequent specific types / niches','',
            '| Reference type | Entities | Examples |','|---|---:|---|']
    for r in niches.head(40).itertuples():
        lines.append(f'| {r.reference_type} | {r.entity_count} | {r.example_entities.replace("|",";")} |')
    lines+=['','The full specific-type table and complete entity inventory are supplied as CSVs. '
        'These figures describe the sample; they do not establish representativeness of all entities or domains.']
    (out/'ENTITY_TABLES.md').write_text('\n'.join(lines)+'\n')
    plot=cats.head(25).iloc[::-1]
    fig,ax=plt.subplots(figsize=(12,10))
    bars=ax.barh(plot.category,plot.entity_count,color='#167a9c')
    ax.bar_label(bars,labels=[str(v) for v in plot.entity_count],padding=4)
    ax.set(xlabel='Unique entities (overlapping gold categories)',
           title='Entity composition: 25 largest gold categories at depth 4',
           xlim=(0,plot.entity_count.max()*1.15))
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('entity_categories.'+ext),dpi=175,bbox_inches='tight')
    plt.close(fig)
    print('Entity tables:',out)

if __name__=='__main__':main()
