"""Primary research tables: graph-aware versus no graph, within each model."""
from pathlib import Path
import argparse
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from plot_stories import load, aligned, CONDITIONS, PAIRS
from evaluation import paired_summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gemini',default='results/gemini_1k_fresh')
    p.add_argument('--llama',default='results/json_mode_final/llama_1k_fast')
    a=p.parse_args()
    rows=[]
    for model,location in [('Gemini',a.gemini),('Llama',a.llama)]:
        d=load(ROOT/location,CONDITIONS)
        for method,(base,graph) in zip(['Standard','CoT','Contrastive'],PAIRS):
            for metric in ['f1','precision','recall','mrr','ndcg']:
                left,right=d[d.condition==base],d[d.condition==graph]
                stat=paired_summary(aligned(left,right,metric))
                rows.append(dict(model=model,method=method,metric=metric,
                    no_graph_mean=left[metric].mean(),graph_aware_mean=right[metric].mean(),**stat))
    out=ROOT/'reports/final_comparison'
    df=pd.DataFrame(rows);df.to_csv(out/'graph_context_all_metrics.csv',index=False)
    lines=['# Primary research comparison: graph-aware versus no graph','',
        'The primary contrast holds model, entities and prompting strategy fixed and adds oracle taxonomy context. '
        'Every comparison contains the same 1,000 entities. Positive deltas favor graph-aware prompting. '
        'Model comparisons are secondary; child removal tests a component of graph context.', '',
        'Scores below are per-entity means. F1 is macro-F1. Intervals are 2,000-resample paired entity '
        'bootstrap intervals, exploratory without multiplicity correction. Printed values use four decimals; CSVs retain full precision.']
    for metric in ['f1','precision','recall','mrr','ndcg']:
        lines+=['','## '+metric.upper(),'',
            '| Model | Prompting | No graph | Graph-aware | Graph gain | 95% paired CI |',
            '|---|---|---:|---:|---:|---|']
        for r in df[df.metric==metric].itertuples():
            lines.append(f'| {r.model} | {r.method} | {r.no_graph_mean:.4f} | {r.graph_aware_mean:.4f} | '
                f'{r.mean_delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |')
    lines+=['','See figures 01 (F1), 16 (precision), 03 (recall), 17 (MRR), 04 (nDCG), '
        'and 08 (paired gains) in both model figure directories. Taxonomy context is oracle context; '
        'these results do not establish performance with imperfect retrieval.']
    (out/'GRAPH_CONTEXT_FOCUS.md').write_text('\n'.join(lines)+'\n')
    print(out/'GRAPH_CONTEXT_FOCUS.md')

if __name__=='__main__':main()
