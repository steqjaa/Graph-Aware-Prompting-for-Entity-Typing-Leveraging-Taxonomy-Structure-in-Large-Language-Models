import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,os
os.makedirs('output/figures',exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
s=pd.read_csv('output/condition_scores.csv');p=pd.read_csv('output/paired_comparisons.csv');e=pd.read_csv('output/entity_scores.csv')
groups=['Gemini','Llama 0','Llama 42','Llama 123'];pairs=[('standard','graph_aware_standard'),('cot_no_graph','cot_graph'),('contrastive_cot_no_graph','contrastive_cot_graph')];styles=['Direct','CoT','Contrastive'];cols=['#526D82','#178078','#B67827','#965580']
def save(name):plt.savefig('output/figures/'+name+'.png',bbox_inches='tight');plt.close()
fig,axs=plt.subplots(2,2,figsize=(10,7.5),sharey=True)
for ax,g in zip(axs.flat,groups):
 ss=s[s.model==g].set_index('condition');x=np.arange(3)
 for k,(lab,col) in enumerate([('No graph','#b9c8d2'),('Graph','#226c89')]):
  v=[ss.loc[t[k],'f1'] for t in pairs];b=ax.bar(x+(k-.5)*.35,v,.35,label=lab,color=col);ax.bar_label(b,fmt='%.4f',padding=3,fontsize=9)
 ax.set_title(g);ax.set_xticks(x,styles);ax.set_ylim(0,.55);ax.set_ylabel('Mean entity F1');ax.legend(frameon=False,loc='upper left')
fig.tight_layout();save('f1_primary')
fig,axs=plt.subplots(1,3,figsize=(10,4),sharey=True)
for k,ax in enumerate(axs):
 ss=p[(p.kind=='graph')&(p.subset=='all')&(p.condition==pairs[k][1])].set_index('model').loc[groups];v=ss.delta_f1.to_numpy();ax.errorbar(v,np.arange(4),xerr=np.vstack([v-ss.low,ss.high-v]),fmt='o',capsize=3,color='#226c89');ax.axvline(0,color='gray',lw=.7)
 for y,(_,r) in enumerate(ss.iterrows()):ax.text(r.delta_f1,y+.18,f'{r.delta_f1:.4f}',ha='center',fontsize=9)
 ax.set_yticks(range(4),groups);ax.set_xlim(-.004,.155);ax.set_title(styles[k]);ax.set_xlabel('Graph minus no graph F1');ax.invert_yaxis()
fig.tight_layout();save('graph_effects')
fig,ax=plt.subplots(figsize=(9,4));cs=['standard','graph_aware_standard','cot_no_graph','cot_graph','contrastive_cot_no_graph','contrastive_cot_graph','standard_parent_only','standard_siblings_only','standard_children_only'];short=['Direct\nNG','Direct\nG','CoT\nNG','CoT\nG','Contr.\nNG','Contr.\nG','Parents','Siblings','Children'];x=np.arange(9)
for k,g in enumerate(groups[1:]):
 v=s[s.model==g].set_index('condition').loc[cs].f1;ax.plot(x,v,'o-',label=g,color=cols[k+1],lw=1)
 for i,y in enumerate(v):ax.annotate(f'{y:.4f}',(i,y),xytext=(0,[-12,9,21][k]),textcoords='offset points',ha='center',fontsize=7)
ax.set_xticks(x,short);ax.set_ylim(.27,.46);ax.set_ylabel('Mean entity F1');ax.legend(ncol=3,frameon=False);fig.tight_layout();save('seeds')
fig,axs=plt.subplots(2,2,figsize=(10,7),sharey=True)
for ax,g in zip(axs.flat,groups):
 ss=s[s.model==g].set_index('condition');x=np.arange(3)
 for k,col in enumerate(['#b9c8d2','#226c89']):
  v=[ss.loc[t[k],'n_types'] for t in pairs];b=ax.bar(x+(k-.5)*.35,v,.35,color=col,label=['No graph','Graph'][k]);ax.bar_label(b,fmt='%.3f',padding=2,fontsize=9)
 ax.set_xticks(x,styles);ax.set_ylim(0,14);ax.set_ylabel('Unique normalized labels per task');ax.set_title(g);ax.legend(frameon=False)
fig.tight_layout();save('length')
fig,ax=plt.subplots(figsize=(8,3.5));ss=p[(p.kind=='child ablation')].set_index('condition').loc[[a[1] for a in pairs]];v=ss.delta_f1.to_numpy();ax.errorbar(v,np.arange(3),xerr=np.vstack([v-ss.low,ss.high-v]),fmt='o',capsize=4,color='#226c89');ax.axvline(0,color='gray');ax.set_yticks(range(3),styles);ax.set_xlabel('Children included minus removed F1');ax.invert_yaxis()
for y,(_,r) in enumerate(ss.iterrows()):ax.text(.013,y,f'{r.delta_f1:.4f} [{r.low:.4f}, {r.high:.4f}]',va='center',fontsize=9)
ax.set_xlim(-.037,.055);fig.tight_layout();save('child_ablation')
# coverage percentages each condition each system as numeric heatmap
cs6=[x for pair in pairs for x in pair];fig,axs=plt.subplots(1,2,figsize=(10,4.8))
for ax,metric,title in zip(axs,['precision','recall'],['Mean gold precision','Mean gold recall']):
 v=s[s.condition.isin(cs6)].pivot(index='condition',columns='model',values=metric).loc[cs6,groups].to_numpy();ax.imshow(v,vmin=0,vmax=.7,cmap='Blues',aspect='auto');ax.set_xticks(range(4),groups,rotation=30,ha='right');ax.set_yticks(range(6),['Direct NG','Direct G','CoT NG','CoT G','Contrastive NG','Contrastive G']);ax.set_title(title)
 for i in range(6):
  for j in range(4):ax.text(j,i,f'{v[i,j]:.4f}',ha='center',va='center',color='white' if v[i,j]>.45 else 'black')
fig.tight_layout();save('coverage')
# pipeline, two rows to fit readable
fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
boxes=[(.02,.57,'Entity name and description\nGold types and taxonomy'),(.37,.57,'Gold-minimal centre\nCapped local relations'),(.72,.57,'Nine frozen prompts\nExternal demonstration'),(.72,.05,'Gemini and three Llama seeds\nValidation and retry log'),(.37,.05,'Aligned per-entity scoring\nFixed and valid-pair cohorts'),(.02,.05,'Prompt and seed comparisons\nCoverage and semantic audit')]
for x,y,t in boxes:ax.text(x+.13,y+.16,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#eff4f6',ec='#526D82'),fontsize=9)
for a,b in [((.28,.73),(.37,.73)),((.63,.73),(.72,.73)),((.85,.58),(.85,.39)),((.72,.21),(.64,.21)),((.37,.21),(.29,.21))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#526D82'))
save('pipeline')
