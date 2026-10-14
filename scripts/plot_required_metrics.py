"""Direct plots for F1, precision, recall, MRR and nDCG from audited scores."""
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot_stories import load, aligned, CONDITIONS, NAMES, PAIRS, BLUE, ORANGE
from evaluation import paired_summary
from figure_labels import label_values

METRICS = [('f1', 'Mean per-entity F1'), ('precision', 'Mean per-entity precision'),
           ('recall', 'Mean per-entity recall'), ('mrr', 'Mean reciprocal rank (MRR)'),
           ('ndcg', 'Mean depth-weighted nDCG')]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gemini', default='results/gemini_1k_fresh')
    p.add_argument('--llama', default='results/json_mode_final/llama_1k_fast')
    p.add_argument('--children', default='results/json_mode_final/llama_children_removed_fast')
    a = p.parse_args()
    g, l = load(ROOT/a.gemini, CONDITIONS), load(ROOT/a.llama, CONDITIONS)
    child = load(ROOT/a.children, [t+'_no_children' for _,t in PAIRS])
    out = ROOT/'reports/final_comparison/required_metric_figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    main_rows, child_rows = [], []
    def save(fig, stem):
        label_values(fig)
        fig.tight_layout()
        for ext in ['png','svg']:
            fig.savefig(out/(stem+'.'+ext), dpi=175, bbox_inches='tight')
        plt.close(fig)
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(11,6))
        y = np.arange(len(CONDITIONS))
        for offset, d, label, color in [(-.18,l,'Llama 3.1 8B Instruct',BLUE),
                                       (.18,g,'Gemini 3.8 Flash',ORANGE)]:
            vals = d.groupby('condition')[metric].mean().reindex(CONDITIONS)
            ax.barh(y+offset,vals,height=.34,color=color,label=label)
            for i,v in enumerate(vals):
                ax.text(v+.008,i+offset,f'{v:.3f}',va='center',fontsize=8)
        for c in CONDITIONS:
            delta=aligned(g[g.condition==c],l[l.condition==c],metric,True)
            main_rows.append(dict(condition=c,metric=metric,
                gemini_mean=g.loc[g.condition==c,metric].mean(),
                llama_mean=l.loc[l.condition==c,metric].mean(),
                **paired_summary(delta)))
        ax.set(yticks=y,yticklabels=NAMES,xlim=(0,1),xlabel=title,
               title=title+' — matched 1,000 entities')
        ax.invert_yaxis();ax.legend()
        save(fig,'models_'+metric)
        fig,ax=plt.subplots(figsize=(9,4.8))
        y=np.arange(3)
        for i,(_,t) in enumerate(PAIRS):
            removed=child[child.condition==t+'_no_children']
            full=l[(l.condition==t)&l.entity_id.isin(removed.entity_id)]
            delta=aligned(removed,full,metric)
            a_mean,b_mean=full[metric].mean(),removed[metric].mean()
            ax.bar(i-.18,a_mean,width=.34,color=BLUE,label='Children included' if i==0 else None)
            ax.bar(i+.18,b_mean,width=.34,color=ORANGE,label='Children removed' if i==0 else None)
            for offset,val in [(-.18,a_mean),(.18,b_mean)]:
                ax.text(i+offset,val+.012,f'{val:.3f}',ha='center',fontsize=9)
            child_rows.append(dict(condition=t,metric=metric,with_children_mean=a_mean,
                                   without_children_mean=b_mean,**paired_summary(delta)))
        ax.set(xticks=y,xticklabels=['Standard','CoT','Contrastive'],ylim=(0,1),
               ylabel=title,title='Llama child context — matched 451 entities')
        ax.legend();save(fig,'children_'+metric)
    pd.DataFrame(main_rows).to_csv(out/'model_metric_comparisons.csv',index=False)
    pd.DataFrame(child_rows).to_csv(out/'child_metric_comparisons.csv',index=False)
    (out/'README.md').write_text('Five direct model comparisons and five matched child comparisons. '
        'Means are per-entity averages; F1 is macro-F1, not the harmonic mean of averaged precision/recall. '
        'MRR follows the original deepest-gold-type ranking evaluator; nDCG uses the original depth-weighted relevance. '
        'Paired CSV intervals use 2,000 entity bootstrap resamples. Model deltas are Llama minus Gemini; '
        'child deltas are children included minus removed. Intervals are exploratory.\n')
    # Keep supplementary Gemini figures numerically labelled.
    for metric, title in [('gold_coverage','Gold coverage (recall)'),
                          ('most_specific_hit','Most-specific hit'),
                          ('mrr','Mean reciprocal rank (MRR)'),
                          ('ndcg','Depth-weighted nDCG'),
                          ('types_count','Mean number of predicted types')]:
        vals=g.groupby('condition')[metric].mean().reindex(CONDITIONS)
        fig,ax=plt.subplots(figsize=(14,6))
        ax.bar(NAMES,vals,color=BLUE)
        ax.tick_params(axis='x',labelrotation=35)
        ax.set(ylabel=title,title='Gemini — '+title)
        label_values(fig);fig.tight_layout()
        for ext in ['png','svg']:
            fig.savefig(ROOT/a.gemini/'analysis'/(metric+'.'+ext),dpi=175,bbox_inches='tight')
        plt.close(fig)
    print('Created 10 required-metric PNG/SVG figure pairs:',out)

if __name__=='__main__':
    main()
