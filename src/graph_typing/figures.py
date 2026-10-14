"""Publication figures from numerical CSVs, with explicit values and cohorts."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS=['Gemini','Llama 0','Llama 42','Llama 123']
PAIRS=[('standard','graph_aware_standard'),('cot_no_graph','cot_graph'),('contrastive_cot_no_graph','contrastive_cot_graph')]
STYLES=['Direct','CoT','Contrastive']

def render(out):
    summary=pd.read_csv(out/'condition_summary.csv')
    comparisons=pd.read_csv(out/'final_comparisons.csv')
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    mapping=[]
    for metric,title in [('f1','Mean entity F1'),('precision','Gold precision'),('recall','Gold recall'),
                         ('mrr','MRR'),('ndcg','nDCG'),('hf1','Hierarchical F1'),('predicted_count','Unique predicted labels')]:
        fig,axs=plt.subplots(2,2,figsize=(11,8),sharey=True)
        top=summary[summary.condition.isin([c for pair in PAIRS for c in pair])][metric].max()*1.24
        for ax,run in zip(axs.flat,RUNS):
            ss=summary[summary.run.eq(run)].set_index('condition')
            for j,(label,color) in enumerate([('No graph','#b7c8d5'),('Graph','#226c89')]):
                values=[ss.loc[pair[j],metric] for pair in PAIRS]
                bars=ax.bar(np.arange(3)+(j-.5)*.36,values,.36,color=color,label=label)
                ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
            ax.set_title(run);ax.set_xticks(np.arange(3),STYLES);ax.set_ylabel(title);ax.set_ylim(0,top);ax.legend(frameon=False)
        fig.suptitle('Completed-condition means; paired effects use matched entities',fontsize=11)
        fig.tight_layout();fig.savefig(folder/f'{metric}.png',dpi=220);plt.close(fig)
        mapping.append(dict(figure=f'{metric}.png',input='condition_summary.csv',metric=metric,cohort='completed per condition'))
    for kind,name,title in [('graph','graph_effects','Graph minus no graph F1'),('child','child_ablation','Children included minus removed F1')]:
        sub=comparisons[comparisons.kind.eq(kind)].copy()
        labels=[f'{r.run}: {r.style} (n={r.n})' for r in sub.itertuples()]
        fig,ax=plt.subplots(figsize=(11,max(3,len(sub)*.48)))
        y=np.arange(len(sub));ax.errorbar(sub.delta,y,xerr=np.vstack([sub.delta-sub.ci_low,sub.ci_high-sub.delta]),fmt='o',capsize=3,color='#226c89')
        for i,r in enumerate(sub.itertuples()):ax.annotate(f'{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]',(r.ci_high,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
        span=max(sub.ci_high.max()-sub.ci_low.min(),.01)
        ax.set_xlim(sub.ci_low.min()-.1*span,sub.ci_high.max()+1.25*span)
        ax.axvline(0,color='gray',lw=.7);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel(title)
        fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=220);plt.close(fig)
        mapping.append(dict(figure=f'{name}.png',input='final_comparisons.csv',filter=kind,cohort='matched completed entities'))
    seed=pd.read_csv(out/'seed_summary.csv')
    ax=seed.plot(x='condition',y=['Llama 0','Llama 42','Llama 123'],kind='bar',figsize=(14,6),width=.8,color=['#178078','#b67827','#965580'])
    for bars in ax.containers:ax.bar_label(bars,fmt='%.4f',padding=2,fontsize=7,rotation=90)
    ax.set_ylim(0,seed[['Llama 0','Llama 42','Llama 123']].max().max()*1.3)
    ax.set_ylabel('Mean entity F1 on common completed cohort');ax.set_xlabel('Condition');plt.xticks(rotation=35,ha='right')
    plt.tight_layout();plt.savefig(folder/'seed_comparison.png',dpi=220);plt.close()
    mapping.append(dict(figure='seed_comparison.png',input='seed_summary.csv',cohort='common completed entities across three seeds'))
    (folder/'manifest.json').write_text(json.dumps(mapping,indent=2)+'\n')
