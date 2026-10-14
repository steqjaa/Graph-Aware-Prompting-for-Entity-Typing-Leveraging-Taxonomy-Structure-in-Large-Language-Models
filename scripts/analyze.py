"""Rebuild pilot figures and diagnostic CSVs from saved results. No inference.
Metric provenance: historical score_ranked_results.py; nDCG family: Jarvelin & Kekalainen (2002).
Depth gains are project-specific. Bootstrap intervals describe entities in this pilot only.
"""
import json, math, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from common import ROOT, ORIGINAL, MAIN_RUN, CONDITIONS, LABELS
import score_ranked_results as scoring
import generate_ranked_prompts as base

A=ROOT/'analysis';F=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none','figure.facecolor':'white','axes.axisbelow':True})
colors=['#526477','#087e8b']*3

def save(fig,name):
    fig.savefig(F/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(F/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def ci(values,seed=33):
    x=np.asarray(values,float);r=np.random.default_rng(seed)
    means=x[r.integers(0,len(x),size=(5000,len(x)))].mean(axis=1)
    return np.quantile(means,[.025,.975])

def pointplot(df,metric,title,xlabel,name,conditions=CONDITIONS,labels=LABELS,limit=(0,1)):
    fig,ax=plt.subplots(figsize=(9,4.5));ys=np.arange(len(conditions));means=[]
    for i,c in enumerate(conditions):
        x=df.loc[df.condition==c,metric].to_numpy(float);m=x.mean();lo,hi=ci(x)
        ax.plot([lo,hi],[i,i],color='#84939b',lw=2)
        ax.scatter(m,i,s=65,color=colors[i%6],zorder=3)
        ax.text(limit[1]+.025,i,f'{m:.3f}',va='center',fontsize=10)
    ax.set(yticks=ys,yticklabels=labels,xlim=limit,xlabel=xlabel,title=title)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.text(.02,.015,'Same 20 entities per condition | 95% entity-bootstrap intervals | historical prompts, not v4',fontsize=8,color='#53636b')
    fig.tight_layout(rect=(0,.05,.96,1));save(fig,name)

def main():
    A.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
    d=pd.read_csv(MAIN_RUN/'weighted_detailed_scores.csv')
    assert not d.duplicated(['entity_id','condition']).any()
    assert d.groupby('condition').size().eq(20).all()
    metadata={json.loads(p.read_text())['entity_id']:json.loads(p.read_text()) for p in (ORIGINAL/'generated_ranked_prompts_v3').glob('example*/metadata.json')}
    raw={}
    for p in MAIN_RUN.glob('example*/*result.json'):
        x=json.loads(p.read_text()); assert x['status']=='completed';raw[x['entity_id'],x['condition']]=x
    verified=0;rankrows=[];fixed=[]
    for _,row in d.iterrows():
        m=metadata[row.entity_id];x=raw[row.entity_id,row.condition];pred=x['predictions'];gold=m['gold_types']
        assert pred==json.loads(row.predictions_json)
        assert gold==json.loads(row.gold_types_json)
        for actual,expected in [(scoring.reciprocal_rank(pred,gold,{}),row.reciprocal_rank),(scoring.ndcg(pred,gold,{}),row.ndcg_all),(scoring.gold_set_metrics(pred,gold,{})['gold_coverage'],row.gold_coverage)]:
            assert abs(actual-expected)<1e-9,(row.entity_id,row.condition,actual,expected)
        verified+=1
        norm=[scoring.normalize_label(t) for t in pred];maxdepth=max(t['depth'] for t in gold)
        for t in gold:
            name=scoring.normalize_label(t['label']);rank=norm.index(name)+1 if name in norm else None
            rankrows.append(dict(entity_id=row.entity_id,entity_label=row.entity_label,condition=row.condition,gold_class_id=t['class_id'],gold_label=t['label'],gold_depth=t['depth'],gold_depth_band='deepest' if t['depth']==maxdepth else 'broader',predicted_rank=rank,status='found' if rank else 'missing',gold_order_note='Depth ties are not ordered relevance judgments'))
        for k in [1,3,5,10]:
            s=scoring.gold_set_metrics(pred[:k],gold,{})
            fixed.append(dict(entity_id=row.entity_id,condition=row.condition,k=k,mrr=scoring.reciprocal_rank(pred,gold,{},k),ndcg=scoring.ndcg(pred,gold,{},k),gold_coverage=s['gold_coverage'],precision_at_k=s['gold_matches_count']/k,returned_in_top_k=len(pred[:k])))
    ranks=pd.DataFrame(rankrows);ranks.to_csv(A/'gold_type_ranks.csv',index=False)
    fx=pd.DataFrame(fixed);fx.to_csv(A/'fixed_k_entity_scores.csv',index=False)
    fx.groupby(['condition','k']).mean(numeric_only=True).to_csv(A/'fixed_k_summary.csv')
    d['unique_predictions']=d.predictions_json.apply(lambda x:len(set(map(scoring.normalize_label,json.loads(x)))))
    d['duplicate_predictions']=d.predictions_count-d.unique_predictions
    lengths=d.groupby('condition').agg(n=('entity_id','size'),mean_types=('predictions_count','mean'),median_types=('predictions_count','median'),min_types=('predictions_count','min'),max_types=('predictions_count','max'),mean_unique=('unique_predictions','mean'),duplicates=('duplicate_predictions','sum'))
    lengths.to_csv(A/'generated_types_per_entity.csv')
    pd.read_csv(MAIN_RUN/'weighted_condition_scores.csv').to_csv(A/'pilot_condition_summary.csv',index=False)
    for metric,title,xlabel,name in [
        ('reciprocal_rank','MRR: first deepest-gold match','Mean reciprocal rank (0-1)','01_mrr'),
        ('ndcg_all','nDCG: graded gold-list ranking','Mean depth-weighted nDCG (0-1)','02_ndcg'),
        ('hit_anywhere','Most-specific recovery: anywhere in the output','Fraction with a deepest-gold match (0-1)','03_most_specific'),
        ('gold_coverage','Gold coverage: all annotated types','Mean entity-level gold recall (0-1)','04_gold_coverage')]:pointplot(d,metric,title,xlabel,name)
    pairs=[('standard','graph_aware_standard'),('cot_no_graph','cot_graph'),('contrastive_cot_no_graph','contrastive_cot_graph'),('standard','retrieved_graph_standard'),('cot_no_graph','cot_retrieved_graph'),('contrastive_cot_no_graph','contrastive_cot_retrieved_graph'),('standard','shuffled_graph_standard')]
    diffs=[]
    for left,right in pairs:
        l=d[d.condition==left].set_index('entity_id');r=d[d.condition==right].set_index('entity_id').loc[l.index]
        for metric in ['reciprocal_rank','ndcg_all','hit_anywhere','gold_coverage','predictions_count']:
            vals=(r[metric]-l[metric]).to_numpy();lo,hi=ci(vals)
            diffs.append(dict(baseline=left,treatment=right,metric=metric,n=len(vals),mean_difference=vals.mean(),ci_low=lo,ci_high=hi,entities_improved=int((vals>0).sum()),entities_worsened=int((vals<0).sum()),entities_tied=int((vals==0).sum())))
    pd.DataFrame(diffs).to_csv(A/'paired_differences.csv',index=False)
    # Oracle and retrieved contexts must remain visibly separate.
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,metric,title in zip(axs,['reciprocal_rank','gold_coverage'],['MRR','Gold coverage']):
        for i,(fam,cs) in enumerate([('Direct',['standard','graph_aware_standard','retrieved_graph_standard']),('CoT',['cot_no_graph','cot_graph','cot_retrieved_graph']),('Contrastive',['contrastive_cot_no_graph','contrastive_cot_graph','contrastive_cot_retrieved_graph'])]):
            vals=[d.loc[d.condition==c,metric].mean() for c in cs];ax.plot(range(3),vals,'o-',label=fam)
        ax.set(xticks=[0,1,2],xticklabels=['No context','Oracle','Retrieved'],ylim=(0,.65),title=title,ylabel='Mean across 20 entities');ax.grid(axis='y',alpha=.15)
    axs[1].legend(frameon=False);fig.suptitle('Context source changes the interpretation',weight='bold');fig.tight_layout();save(fig,'05_context_source')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for i,c in enumerate(CONDITIONS):
        x=d.loc[d.condition==c,'predictions_count'].to_numpy();jitter=np.linspace(-.12,.12,len(x))
        ax.scatter(x,i+jitter,s=22,alpha=.6,color=colors[i]);ax.scatter(x.mean(),i,marker='D',s=65,color='#152e3d',zorder=4)
        ax.text(max(d.predictions_count)+1,i,f'{x.mean():.2f}',va='center')
    ax.set(yticks=range(6),yticklabels=LABELS,xlabel='Parsed type labels per entity (dots); mean (diamond)',title='Output length is part of the result',xlim=(0,max(d.predictions_count)+3));ax.invert_yaxis();ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'06_output_lengths')
    # Full heatmap across gold entity/type pairs; M is absence, never a fabricated rank.
    rr=ranks[ranks.condition.isin(CONDITIONS)]
    indexcols=['entity_id','entity_label','gold_depth','gold_class_id','gold_label']
    pivot=rr.pivot(index=indexcols,columns='condition',values='predicted_rank').reindex(columns=CONDITIONS)
    pivot=pivot.sort_index(level=[1,2,4],ascending=[True,False,True])
    for page,start in enumerate(range(0,len(pivot),32),1):
        sub=pivot.iloc[start:start+32];values=sub.to_numpy();fig,ax=plt.subplots(figsize=(12,max(5,len(sub)*.26+1.8)))
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#efefef')
        im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=1,vmax=20)
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,'M' if np.isnan(v) else str(int(v)),ha='center',va='center',fontsize=8,color='white' if v>11 else '#19323f')
        ax.set(xticks=range(6),xticklabels=LABELS,yticks=range(len(sub)),yticklabels=[textwrap.shorten(f'{x[1]} | {x[4]} (d{x[2]})',width=66,placeholder='...') for x in sub.index],title=f'Gold-type output ranks | page {page} | M = not returned')
        ax.tick_params(axis='x',rotation=25);ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,label='Actual output position (lower is earlier)',shrink=.7);fig.tight_layout();save(fig,f'07_gold_ranks_{page:02d}')
    # Type-level hit rate heatmap with explicit support counts.
    types=rr.assign(hit=rr.status.eq('found').astype(int)).groupby(['gold_class_id','gold_label','condition']).agg(recall=('hit','mean'),support=('entity_id','nunique')).reset_index()
    types.to_csv(A/'type_recall_support.csv',index=False)
    tp=types.pivot(index=['gold_class_id','gold_label'],columns='condition',values='recall').reindex(columns=CONDITIONS).sort_index(level=1)
    supports=types.groupby(['gold_class_id','gold_label']).support.first()
    for page,start in enumerate(range(0,len(tp),32),1):
        sub=tp.iloc[start:start+32];fig,ax=plt.subplots(figsize=(10,max(5,len(sub)*.25+1.8)))
        im=ax.imshow(sub,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        for (i,j),v in np.ndenumerate(sub.to_numpy()):ax.text(j,i,f'{v:.0%}',ha='center',va='center',fontsize=8,color='white' if v>.6 else '#19323f')
        ax.set(xticks=range(6),xticklabels=LABELS,yticks=range(len(sub)),yticklabels=[f'{x[1]} (n={supports.loc[x]})' for x in sub.index],title=f'Gold-type recall | page {page} | support = annotated entities')
        ax.tick_params(axis='x',rotation=25);ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,label='Fraction recovered',shrink=.7);fig.tight_layout();save(fig,f'08_type_recall_{page:02d}')
    misses=[]
    for (eid,gid),group in rr.groupby(['entity_id','gold_class_id']):
        by=group.set_index('condition')
        for l,r in pairs[:3]:
            x,y=by.loc[l],by.loc[r]
            if x.status=='missing' or y.status=='missing':misses.append(dict(entity_id=eid,entity_label=x.entity_label,gold_class_id=gid,gold_label=x.gold_label,gold_depth=x.gold_depth,baseline=l,treatment=r,baseline_rank=x.predicted_rank,treatment_rank=y.predicted_rank,transition=f'{x.status}->{y.status}'))
    pd.DataFrame(misses).to_csv(A/'missing_type_transitions.csv',index=False)
    # Relation availability and same-run paired sensitivity (oracle full vs baseline).
    relation=[]
    for eid,m in metadata.items():
        relation.append(dict(entity_id=eid,**{k+'_available':m['graph_context'][k] not in ('None','',None) for k in ['parents','siblings','children']}))
    rel=pd.DataFrame(relation);rel.to_csv(A/'pilot_relation_availability.csv',index=False)
    strat=[]
    for label,ids in [('all',set(rel.entity_id)),('children_present',set(rel[rel.children_available].entity_id)),('children_absent',set(rel[~rel.children_available].entity_id))]:
        for l,r in pairs[:3]:
            x=d[(d.condition==l)&d.entity_id.isin(ids)].set_index('entity_id');y=d[(d.condition==r)&d.entity_id.isin(ids)].set_index('entity_id').loc[x.index]
            for metric in ['reciprocal_rank','ndcg_all','hit_anywhere','gold_coverage']:
                vals=(y[metric]-x[metric]).to_numpy();lo,hi=ci(vals)
                strat.append(dict(stratum=label,baseline=l,treatment=r,n=len(x),metric=metric,mean_difference=vals.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(strat).to_csv(A/'children_availability_sensitivity.csv',index=False)
    # Do NOT mix separate-backend relation ablations with the bf16 run.
    abl=ORIGINAL/'generated_ranked_results/_cf_meta_llama-3.1-8b-instruct-fast__graph-ablation-local-mps-v1/weighted_detailed_scores.csv'
    ad=pd.read_csv(abl);ad.groupby('condition')[['reciprocal_rank','ndcg_all','hit_anywhere','gold_coverage','predictions_count']].mean().to_csv(A/'separate_backend_ablation_descriptive_only.csv')
    (A/'validation.json').write_text(json.dumps(dict(raw_results_recomputed=verified,conditions=len(d.condition.unique()),entities=d.entity_id.nunique(),main_run=str(MAIN_RUN.relative_to(ROOT)),matching='historical normalized labels, no aliases',bootstrap_replicates=5000,bootstrap_seed=33,models_rerun=False,main_six=CONDITIONS),indent=2))
    print('Verified saved metrics against',verified,'raw results; figures:',len(list(F.glob('*.png'))))
if __name__=='__main__':main()
