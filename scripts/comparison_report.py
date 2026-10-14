"""Write an exact table/report from scored results; no inference or invented outcomes."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot_stories import load, CONDITIONS, LABEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gemini', default='results/gemini_1k_fresh')
    p.add_argument('--llama')
    p.add_argument('--children')
    p.add_argument('--output', default='reports/final_comparison')
    a = p.parse_args()
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    summary, inventory = [], []
    runs = [('Gemini 3.8 Flash', a.gemini)]
    if a.llama:
        runs.append(('Llama 3.1 8B Instruct (vLLM)', a.llama))
    for label, location in runs:
        root = ROOT / location
        d = load(root, CONDITIONS)
        s = pd.read_csv(root / 'analysis/summary.csv')
        s.insert(0, 'model_system', label)
        summary.append(s)
        inventory.append(dict(model=label, responses=len(d), entities=d.entity_id.nunique(), result_dir=location))
    combined = pd.concat(summary, ignore_index=True)
    combined.to_csv(out / 'model_condition_metrics.csv', index=False)
    lines = ['# Final experiment evaluation', '',
        'The primary research contrast is graph-aware versus no-graph prompting within each model. See GRAPH_CONTEXT_FOCUS.md for all five metrics and paired gains. Model comparisons and child ablations are supporting analyses.', '',
        'Scores below use completed, matched main-run results. Taxonomy context is oracle context.', '',
        '| Model system | Condition | n | Macro-F1 | Micro-F1 | Hierarchical F1 | Most-specific hit |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in combined.to_dict('records'):
        lines.append(f"| {r['model_system']} | {LABEL[r['condition']]} | {int(r['n'])} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} | {r['hierarchical_f1']:.4f} | {r['most_specific_hit']:.4f} |")
    lines += ['', 'Hierarchical F1 uses ancestor closure, including self and roots; it is not LCA-F1.',
              'Gemini and Llama use different inference systems and decoding settings. The comparison does not isolate architecture.',
              'Confidence intervals are paired entity bootstraps (2,000 resamples); effects are exploratory, without multiplicity correction.', '']
    if not a.llama:
        lines.append('Llama main and child-removal inference have not been run here. No Llama comparison is reported.')
    else:
        figures = ROOT / a.llama / 'analysis/story_figures'
        for stem, title in [('10_model_paired_effects', 'Llama minus Gemini: paired entity F1'),
                            ('11_model_graph_uplift', 'Difference in taxonomy uplift: Llama minus Gemini'),
                            ('12_matched_children', 'Children included minus children removed')]:
            f = figures / (stem + '.csv')
            if f.exists():
                frame = pd.read_csv(f)
                frame.to_csv(out / (stem + '.csv'), index=False)
                lines += ['', '## ' + title, '', '| Comparison | Metric | n | Mean delta | 95% CI |', '|---|---|---:|---:|---|']
                for r in frame.to_dict('records'):
                    lines.append(f"| {r['label']} | {r['metric']} | {int(r['n'])} | {r['mean_delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    lines += ['', 'Child-removal comparisons use only the 451 child-present entities and reuse their full-context main-run responses.',
              'Older incomplete Gemini outputs and the 20-entity Llama pilot are historical runs, not pooled into the final comparison.',
              'Prediction-error review worksheets require real human annotation; outside-taxonomy labels are not automatically hallucinations.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    (out / 'run_inventory.json').write_text(json.dumps(inventory, indent=2))
    print(out)


if __name__ == '__main__':
    main()
