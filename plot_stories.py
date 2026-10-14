"""Thesis figure suite: flat/hierarchical F1 first, rankings as supporting analysis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from evaluation import PAIRS, paired_summary
from figure_labels import label_values, numeric_heatmap

CONDITIONS = ['standard', 'graph_aware_standard', 'cot_no_graph', 'cot_graph',
              'contrastive_cot_no_graph', 'contrastive_cot_graph',
              'standard_parent_only', 'standard_siblings_only', 'standard_children_only']
NAMES = ['Standard', 'Standard + taxonomy', 'CoT', 'CoT + taxonomy', 'Contrastive',
         'Contrastive + taxonomy', 'Parent only', 'Sibling only', 'Children only']
LABEL = dict(zip(CONDITIONS, NAMES))
BLUE, ORANGE = '#167a9c', '#ed9a35'

def load(root, conditions):
    analysis = root / 'analysis'
    info = json.loads((analysis / 'interpretation.json').read_text())
    if info.get('schema_version') != 2:
        raise ValueError(f'Rescore {root} with the updated scorer first')
    completion = pd.read_csv(analysis / 'completion_status.csv')
    if not completion.status.eq('completed').all():
        raise ValueError('Latest scoring attempt was incomplete; refusing stale plots')
    d = pd.read_csv(analysis / 'entity_scores.csv')
    if len(completion) != len(d) or set(zip(completion.entity_id, completion.condition)) != set(zip(d.entity_id, d.condition)):
        raise ValueError('Completion audit and scores differ')
    if d.duplicated(['entity_id', 'condition']).any() or set(d.condition) != set(conditions):
        raise ValueError('Duplicate pairs or unexpected conditions')
    if len(d) != len(conditions) * d.entity_id.nunique():
        raise ValueError('Incomplete entity x condition cohort')
    if d.groupby('entity_id').gold_signature.nunique().max() != 1:
        raise ValueError('Gold sets differ across conditions')
    return d

def aligned(left, right, metric, verify_input=False):
    left, right = left.set_index('entity_id'), right.set_index('entity_id')
    if set(left.index) != set(right.index):
        raise ValueError('Paired comparison cohorts differ; refusing silent intersection')
    right = right.reindex(left.index)
    if not left.gold_signature.equals(right.gold_signature):
        raise ValueError('Paired gold annotations differ')
    if verify_input and not left.input_hash.equals(right.input_hash):
        raise ValueError('Model comparison prompts/metadata differ')
    return right[metric] - left[metric]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results/gemini_1k_fresh')
    p.add_argument('--primary-label', default='Primary model')
    p.add_argument('--compare-results')
    p.add_argument('--compare-label', default='Comparison model')
    p.add_argument('--children-results')
    a = p.parse_args()
    root = Path(a.results).resolve()
    d = load(root, CONDITIONS)
    if d.entity_id.nunique() != 1000:
        p.error('Expected all 1000 entities for the main experiment')
    other = load(Path(a.compare_results), CONDITIONS) if a.compare_results else None
    child_conditions = [g + '_no_children' for _, g in PAIRS]
    child = load(Path(a.children_results), child_conditions) if a.children_results else None
    if other is not None:
        for c in CONDITIONS:
            aligned(d[d.condition == c], other[other.condition == c], 'f1', True)
    if child is not None:
        eligible = d[d.children_available.astype(str).str.lower().isin(['true', '1'])]
        for _, g in PAIRS:
            full = eligible[eligible.condition == g]
            removed = child[child.condition == g + '_no_children']
            aligned(full, removed, 'f1')
            check = removed.set_index('entity_id').reindex(full.entity_id)
            if not np.array_equal(full.prompt_sha256.values, check.source_prompt_sha256.values):
                p.error('Child ablation source prompts do not match main prompts')
    out = root / 'analysis/story_figures'
    out.mkdir(parents=True, exist_ok=True)
    # Remove only obsolete figures produced by the previous version.
    for stem in ['01_radial_mrr', '02_mrr_survival', '03_graph_gain_distributions',
                 '04_paired_metric_effects', '05_gold_rank_transitions', '06_depth_recall',
                 '07_output_tradeoff', '08_child_availability_effect', '09_type_recall_shifts_all', '09_type_recall_shifts']:
        for ext in ['png', 'svg']:
            (out / f'{stem}.{ext}').unlink(missing_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    figures, effects = [], []
    def save(fig, name, caption):
        label_values(fig)
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(out / f'{name}.{ext}', dpi=175, bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(name=name, description=caption))
    def groups(metric, title, name):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for i, (base, graph) in enumerate(PAIRS):
            for offset, c, color, label in [(-.18, base, BLUE, 'No graph'), (.18, graph, ORANGE, 'Graph-aware (taxonomy)')]:
                value = d.loc[d.condition == c, metric].mean()
                ax.bar(i + offset, value, width=.34, color=color, label=label if i == 0 else None)
                ax.text(i + offset, value + .014, f'{value:.3f}', ha='center', fontsize=9)
        ax.set(xticks=range(3), xticklabels=['Standard', 'CoT', 'Contrastive'], ylim=(0, 1), ylabel=title,
               title=f'{a.primary_label}: {title}')
        ax.legend()
        save(fig, name, 'Same 1000 entities per condition; descriptive means. Paired uncertainty is reported separately.')
    groups('f1', 'Mean per-entity F1', '01_macro_f1')
    groups('hierarchical_f1', 'Hierarchical F1', '02_hierarchical_f1')
    groups('precision', 'Mean per-entity precision', '16_graph_precision')
    groups('mrr', 'Mean reciprocal rank (MRR)', '17_graph_mrr')
    groups('gold_coverage', 'Gold coverage (recall)', '03_gold_coverage')
    groups('ndcg', 'Depth-weighted nDCG (supporting)', '04_ndcg_supporting')
    order = ['standard', 'standard_parent_only', 'standard_siblings_only', 'standard_children_only', 'graph_aware_standard']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh([LABEL[c] for c in order], d.groupby('condition').f1.mean().reindex(order), color=BLUE)
    ax.invert_yaxis(); ax.set(xlim=(0, 1), xlabel='Mean per-entity F1', title='Which taxonomy context helps?')
    save(fig, '05_context_ablation', 'Relation-only and full contexts have different node budgets; oracle context, descriptive comparison.')
    bins = ['Rank 1', 'Rank 2', 'Rank 3+', 'Miss']
    table = []
    for c in CONDITIONS[:6]:
        rr = d.loc[d.condition == c, 'mrr'].to_numpy()
        table.append([np.mean(rr == 1), np.mean(rr == .5), np.mean((rr > 0) & (rr < .5)), np.mean(rr == 0)])
    tab = pd.DataFrame(table, index=CONDITIONS[:6], columns=bins)
    tab.to_csv(out / 'rank_distribution.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    numeric_heatmap(ax, tab.rename(index=LABEL), 'Rank of first deepest-gold type (supporting)', 'Fraction of entities; rows sum to 1')
    save(fig, '06_rank_distribution', 'First deepest annotated type: rank 1, rank 2, rank 3+, or not recovered.')
    depth_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric in zip(axes, ['f1', 'gold_coverage']):
        for offset, color, (base, graph) in zip([-.17, 0, .17], [BLUE, ORANGE, '#138568'], PAIRS):
            l, r = d[d.condition == base], d[d.condition == graph]
            dx = aligned(l, r, metric)
            depth = l.set_index('entity_id').target_depth
            for i, (label, mask) in enumerate([('4–5', depth.between(4, 5)), ('6–7', depth.between(6, 7)), ('8+', depth >= 8)]):
                values = dx[mask.reindex(dx.index)]
                if values.empty:
                    continue
                stat = paired_summary(values)
                depth_rows.append(dict(baseline=base, treatment=graph, metric=metric, depth=label, **stat))
                ax.plot([i+offset]*2, [stat['ci_low'], stat['ci_high']], color=color)
                ax.scatter(i+offset, stat['mean_delta'], color=color, label=LABEL[base] if i == 0 else None)
        ax.axhline(0, color='gray', lw=1)
        ax.set(xticks=range(3), xticklabels=['4–5', '6–7', '8+'], xlabel='Maximum annotated target depth',
               ylabel='Graph-aware − no graph', title='Per-entity F1' if metric == 'f1' else 'Gold coverage')
        ax.legend()
    save(fig, '07_uplift_by_target_depth', 'One paired observation per entity. Bars: 95% paired bootstrap CI. Unsupported depth buckets omitted; all supports in CSV.')
    pd.DataFrame(depth_rows).to_csv(out / 'depth_uplift.csv', index=False)
    (out / 'depth_support.json').write_text(json.dumps(dict(below_4=int((d[d.condition == 'standard'].target_depth < 4).sum()))))
    def forest(rows, name, title):
        data = pd.DataFrame(rows)
        for raw, pretty in [('hierarchical_f1','hierarchical F1'),('most_specific_hit','most-specific hit'),('gold_coverage','recall'),('f1','F1'),('mrr','MRR'),('ndcg','nDCG')]:
            data['label'] = data['label'].str.replace(' · '+raw, ' · '+pretty, regex=False)
        fig, ax = plt.subplots(figsize=(12, max(5, .52 * len(data) + 1.8)))
        for i, row in data.iterrows():
            ax.plot([row.ci_low, row.ci_high], [i, i], color=BLUE, lw=2)
            ax.scatter(row.mean_delta, i, color=ORANGE, zorder=3)
        ax.axvline(0, color='gray', lw=1)
        ax.set(yticks=range(len(data)), yticklabels=data.label, xlabel='Mean paired difference (95% bootstrap CI)', title=title)
        ax.invert_yaxis()
        save(fig, name, '2000 paired entity bootstrap resamples; fixed seed 230917; exploratory intervals without multiplicity correction.')
        data.to_csv(out / f'{name}.csv', index=False)
        effects.extend(rows)
    rows = []
    for base, graph in PAIRS:
        for metric in ['f1', 'precision', 'recall', 'mrr', 'ndcg', 'hierarchical_f1', 'most_specific_hit']:
            stat = paired_summary(aligned(d[d.condition == base], d[d.condition == graph], metric))
            rows.append(dict(label=f'{LABEL[base]} · {metric}', baseline=base, treatment=graph, metric=metric, **stat))
    forest(rows, '08_paired_graph_effects', 'Graph-aware − no graph')
    if other is not None:
        rows = []
        fig, ax = plt.subplots(figsize=(11, 6))
        y = np.arange(9)
        for offset, data, label, color in [(-.18, d, a.primary_label, BLUE), (.18, other, a.compare_label, ORANGE)]:
            ax.barh(y+offset, data.groupby('condition').f1.mean().reindex(CONDITIONS), height=.34, label=label, color=color)
        ax.set(yticks=y, yticklabels=NAMES, xlim=(0, 1), xlabel='Mean per-entity F1', title='Model-system comparison')
        ax.invert_yaxis(); ax.legend()
        save(fig, '09_model_comparison', 'Matched prompts and gold labels. Decoding, provider thinking, and model differ; not an architecture-only comparison.')
        for c in CONDITIONS:
            stat = paired_summary(aligned(other[other.condition == c], d[d.condition == c], 'f1', True))
            rows.append(dict(label=LABEL[c], baseline=a.compare_label, treatment=a.primary_label, condition=c, metric='f1', **stat))
        forest(rows, '10_model_paired_effects', f'{a.primary_label} − {a.compare_label}')
        rows = []
        for base, graph in PAIRS:
            dx = aligned(d[d.condition == base], d[d.condition == graph], 'f1')
            dy = aligned(other[other.condition == base], other[other.condition == graph], 'f1')
            stat = paired_summary(dx - dy.reindex(dx.index))
            rows.append(dict(label=LABEL[base], baseline=a.compare_label, treatment=a.primary_label, metric='f1_uplift', **stat))
        forest(rows, '11_model_graph_uplift', 'Difference in graph uplift: primary − comparison')
    if child is not None:
        rows = []
        for _, graph in PAIRS:
            removed = child[child.condition == graph + '_no_children']
            full = d[(d.condition == graph) & d.entity_id.isin(removed.entity_id)]
            for metric in ['f1', 'hierarchical_f1', 'gold_coverage', 'most_specific_hit']:
                stat = paired_summary(aligned(removed, full, metric))
                rows.append(dict(label=f'{LABEL[graph]} · {metric}', baseline=graph+'_no_children', treatment=graph, metric=metric, **stat))
        forest(rows, '12_matched_children', f'Children included − children removed (n={child.entity_id.nunique()})')
    pr = pd.read_csv(root / 'analysis/prediction_errors.csv')
    categories = ['Exact gold', '1 hop', '2 hops', '3+ hops', 'Outside WiKC', 'Unresolved/disconnected']
    def bucket(row):
        if row.correct:
            return 'Exact gold'
        if row.category == 'out_of_taxonomy':
            return 'Outside WiKC'
        if pd.isna(row.distance):
            return 'Unresolved/disconnected'
        return '1 hop' if row.distance == 1 else ('2 hops' if row.distance == 2 else '3+ hops')
    pr['distance_bucket'] = pr.apply(bucket, axis=1)
    count = pd.crosstab(pr.condition, pr.distance_bucket).reindex(index=CONDITIONS, columns=categories, fill_value=0)
    shares = count.div(count.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 6))
    numeric_heatmap(ax, shares.rename(index=LABEL), 'Taxonomy error distance', 'Share of unique predicted labels; rows sum to 1')
    save(fig, '13_taxonomy_error_distance', 'Distance to nearest gold class on undirected WiKC; no invented distance for unresolved predictions.')
    count.to_csv(out / 'distance_bucket_counts.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(NAMES, d.groupby('condition').hierarchy_consistent_set.mean().reindex(CONDITIONS), color=BLUE)
    ax.invert_yaxis(); ax.set(xlim=(0, 1), xlabel='Fraction of eligible entities', title='All resolved predicted nodes include their ancestors')
    save(fig, '14_hierarchy_consistency', 'Undefined for entities with no resolved graph nodes; roots included; consult eligible denominators in summary.csv.')
    # Directional error composition, all errors as denominator (including unresolved).
    error = pr[pr.correct == 0]
    direction = pd.crosstab(error.condition, error.error_bucket).reindex(index=CONDITIONS, fill_value=0)
    fractions = direction.div(direction.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(max(14, len(fractions.columns)*1.05), 7))
    numeric_heatmap(ax, fractions.rename(index=LABEL), 'Direction of taxonomy errors', 'Share of incorrect unique predictions; rows sum to 1')
    save(fig, '15_directional_errors', 'Directed distance for broader/narrower; shared immediate parent for siblings. Precedence and unresolved cases documented in interpretation.json.')
    direction.to_csv(out / 'directional_error_counts.csv')
    pd.DataFrame(effects).to_csv(out / 'all_paired_effects.csv', index=False)
    (root / 'analysis/story_plot_manifest.json').write_text(json.dumps(dict(figures=figures, entities=1000,
        primary_label=a.primary_label, compare_label=a.compare_label if other is not None else None,
        model_comparison='Model systems differ in decoding/provider thinking; architecture is not isolated',
        primary_metrics=['macro_f1','micro_f1','hierarchical_f1','most_specific_hit'],
        secondary_metrics=['mrr','ndcg']), indent=2))
    print(f'Created {len(figures)} PNG/SVG figure pairs in {out}')

if __name__ == '__main__':
    main()
