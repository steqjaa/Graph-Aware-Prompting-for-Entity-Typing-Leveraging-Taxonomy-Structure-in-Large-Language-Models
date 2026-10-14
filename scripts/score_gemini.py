"""Model-agnostic manifest scoring. Refuses incomplete or mismatched experiments."""
import argparse
import csv
import hashlib
import json
import pandas as pd
from common import ROOT
import score_ranked_results as legacy
from evaluation import Taxonomy, PAIRS, paired_summary
from run_llama import parse

METRICS = ['f1', 'precision', 'recall', 'hierarchical_f1', 'hierarchical_precision',
           'hierarchical_recall', 'most_specific_hit', 'strict_accuracy',
           'hierarchy_consistency', 'hierarchy_consistent_set', 'taxonomy_error_mean',
           'mrr', 'ndcg', 'gold_coverage', 'gold_precision', 'types_count']

def summarize(d):
    result = d.groupby('condition')[METRICS].mean().rename(columns={'f1':'macro_f1'})
    result.insert(0, 'n', d.groupby('condition').size())
    totals = d.groupby('condition')[['tp', 'predicted_count', 'gold_count']].sum()
    result['micro_precision'] = totals.tp / totals.predicted_count.replace(0, float('nan'))
    result['micro_precision'] = result['micro_precision'].fillna(0)
    result['micro_recall'] = totals.tp / totals.gold_count
    result['micro_f1'] = 2 * totals.tp / (totals.predicted_count + totals.gold_count)
    result['loose_macro_f1'] = (2 * result.precision * result.recall / (result.precision + result.recall)).fillna(0)
    result['consistency_eligible_entities'] = d.groupby('condition').hierarchy_consistency.count()
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='prompts_v5/all_1k/manifest.csv')
    p.add_argument('--results', default='results/gemini_1k')
    p.add_argument('--manual-sample-size', type=int, default=20, help='Per condition and error category')
    a = p.parse_args()
    if a.manual_sample_size < 1:
        p.error('manual-sample-size must be positive')
    rows = list(csv.DictReader((ROOT / a.manifest).open()))
    if not rows or len({(r['entity_id'], r['condition']) for r in rows}) != len(rows):
        p.error('Empty manifest or duplicate pairs')
    out = ROOT / a.results
    dest = out / 'analysis'
    dest.mkdir(parents=True, exist_ok=True)
    inputs, availability, expected = [], [], set()
    for row in rows:
        path = out / f"example_{int(row['example_number']):04d}" / f"{row['condition']}_result.json"
        if path in expected:
            p.error('Manifest output path collision')
        expected.add(path)
        x = json.loads(path.read_text()) if path.exists() else {'status':'missing'}
        status = x.get('status', 'missing')
        if status == 'completed':
            meta = json.loads((ROOT / row['metadata_path']).read_text())
            prompt = (ROOT / row['prompt_path']).read_text()
            fp = hashlib.sha256((prompt + json.dumps(meta, sort_keys=True)).encode()).hexdigest()
            if (x.get('entity_id') != row['entity_id'] or x.get('condition') != row['condition']
                    or x.get('metadata') != meta or x.get('input_hash') != fp):
                p.error(f'Result identity/input mismatch: {path}')
            pred = x.get('predictions')
            if not isinstance(pred, list) or not pred or any(not isinstance(v, str) or not v.strip() for v in pred):
                status = 'invalid_predictions'
            parsed_status, parsed_predictions, parsed_explanation = parse(
                x.get('response', ''), row['condition'], x.get('finish_reason') == 'STOP')
            if parsed_status != 'completed':
                status = parsed_status
            elif parsed_predictions != pred or parsed_explanation != x.get('explanation', ''):
                status = 'response_parse_mismatch'
            inputs.append((row, x))
        availability.append(dict(entity_id=row['entity_id'], condition=row['condition'], status=status))
    if set(out.glob('example_*/*_result.json')) - expected:
        p.error('Unexpected results outside manifest')
    av = pd.DataFrame(availability)
    av.to_csv(dest / 'completion_status.csv', index=False)
    print(av.groupby(['condition', 'status']).size().to_string(), flush=True)
    if not av.status.eq('completed').all():
        raise SystemExit('Incomplete run; no score averages generated. Resume inference/retry failed jobs.')
    taxonomy = Taxonomy()
    if not taxonomy.is_dag:
        p.error('WiKC contains cycles; hierarchy evaluation requires an acyclic graph')
    records, ranks, predictions = [], [], []
    for row, x in inputs:
        m, pred = x['metadata'], x['predictions']
        gold = m['gold_types']
        metrics, details = taxonomy.score(pred, gold)
        r = dict(entity_id=row['entity_id'], entity_label=row['entity_label'], condition=row['condition'],
                 children_available=m['children_available'], demonstration_omitted=m['demonstration_omitted'],
                 input_hash=x['input_hash'], gold_signature=hashlib.sha256(json.dumps(m['gold_types'], sort_keys=True).encode()).hexdigest(),
                 source_prompt_sha256=row.get('source_prompt_sha256', ''), prompt_sha256=hashlib.sha256((ROOT / row['prompt_path']).read_bytes()).hexdigest(),
                 model_version=x.get('model_version', x.get('model', 'unknown')),
                 mrr=legacy.reciprocal_rank(pred, gold, {}), ndcg=legacy.ndcg(pred, gold, {}),
                 most_specific_hit=legacy.hit_at_k(pred, gold, {}), types_count=len(pred), **metrics)
        r.update(gold_coverage=r['recall'], gold_precision=r['precision'])
        for k in [1, 3, 5, 10]:
            r[f'coverage_at_{k}'] = legacy.gold_set_metrics(pred[:k], gold, {})['gold_coverage']
            r[f'ndcg_at_{k}'] = legacy.ndcg(pred, gold, {}, k)
        records.append(r)
        normalized = [legacy.normalize_label(v) for v in pred]
        for t in gold:
            label = legacy.normalize_label(t['label'])
            ranks.append(dict(entity_id=row['entity_id'], condition=row['condition'], gold_class_id=t['class_id'],
                              gold_type=t['label'], gold_depth=t['depth'], rank=normalized.index(label)+1 if label in normalized else None))
        for detail in details:
            predictions.append(dict(entity_id=row['entity_id'], entity_label=row['entity_label'], description=m['description'],
                condition=row['condition'], gold_labels=' | '.join(t['label'] for t in gold), **detail))
    d, pr = pd.DataFrame(records), pd.DataFrame(predictions)
    d.to_csv(dest / 'entity_scores.csv', index=False)
    pd.DataFrame(ranks).to_csv(dest / 'gold_type_ranks.csv', index=False)
    pr.to_csv(dest / 'prediction_errors.csv', index=False)
    for name, sub in [('all_1000', d), ('few_shot_only', d[~d.demonstration_omitted]), ('children_present', d[d.children_available])]:
        summarize(sub).to_csv(dest / f'summary_{name}.csv')
    summarize(d).to_csv(dest / 'summary.csv')
    distribution = pr.groupby(['condition', 'error_bucket']).size().rename('count').reset_index()
    distribution['share_all_predictions'] = distribution['count'] / distribution.condition.map(pr.groupby('condition').size())
    distribution.to_csv(dest / 'error_distribution.csv', index=False)
    errors = pr[pr.correct.eq(0)]
    distance_summary = errors.groupby('condition').distance.agg(['count', 'mean', 'median'])
    distance_summary['incorrect_predictions'] = errors.groupby('condition').size()
    distance_summary['unmeasurable'] = distance_summary.incorrect_predictions - distance_summary['count']
    distance_summary.to_csv(dest / 'taxonomy_error_distance.csv')
    pairs, examples = [], []
    comparisons = PAIRS + [('standard_parent_only', 'standard_siblings_only'),
                           ('standard_parent_only', 'standard_children_only'),
                           ('standard_siblings_only', 'standard_children_only')]
    for base, treatment in comparisons:
        if not {base, treatment} <= set(d.condition):
            continue
        for metric in ['f1', 'precision', 'recall', 'hierarchical_f1', 'most_specific_hit', 'mrr', 'ndcg']:
            wide = d[d.condition.isin([base, treatment])].pivot(index='entity_id', columns='condition', values=metric)
            if wide.isna().any().any():
                p.error('Incomplete paired cohort')
            delta = wide[treatment] - wide[base]
            pairs.append(dict(baseline=base, treatment=treatment, metric=metric, **paired_summary(delta)))
            if metric == 'f1':
                for direction, values in [('helped', delta[delta > 1e-12].nlargest(5)), ('harmed', delta[delta < -1e-12].nsmallest(5))]:
                    for eid, value in values.items():
                        example = dict(entity_id=eid, baseline=base, treatment=treatment, direction=direction, delta_f1=value)
                        for side, cond in [('baseline', base), ('treatment', treatment)]:
                            detail = pr[(pr.entity_id == eid) & (pr.condition == cond)]
                            example[side + '_predictions'] = ' | '.join(detail.prediction)
                        detail = pr[pr.entity_id == eid].iloc[0]
                        example.update(entity_label=detail.entity_label, description=detail.description, gold_labels=detail.gold_labels)
                        examples.append(example)
    pd.DataFrame(pairs).to_csv(dest / 'paired_effects.csv', index=False)
    pd.DataFrame(examples).to_csv(dest / 'representative_cases.csv', index=False)
    samples = [group.sample(min(len(group), a.manual_sample_size), random_state=230917)
               for _, group in errors.groupby(['condition', 'category'])]
    sample = pd.concat(samples, ignore_index=True) if samples else errors.copy()
    for column in ['valid_type_yes_no_uncertain', 'evidence_source', 'annotator', 'notes']:
        sample[column] = ''
    # Preserve human annotations on repeat scoring.
    review = dest / 'manual_validation_sample.csv'
    if not review.exists():
        sample.to_csv(review, index=False)
    definitions = dict(schema_version=2, entities=d.entity_id.nunique(), completed=len(d),
        model_versions=sorted(d.model_version.unique().tolist()), hierarchy='Ancestor closure including self and roots; not LCA-F1',
        macro_f1='Mean entity F1; loose_macro_f1 is harmonic mean of mean entity precision and recall',
        micro_f1='2 sum TP / (sum predicted labels + sum gold labels)',
        duplicates='Normalized labels deduplicated for set metrics; historical ranking metrics unchanged',
        ambiguity='Exact gold labels resolve to annotated IDs; otherwise only unique label mappings resolve; unresolved labels penalize hierarchical precision',
        distance='Minimum undirected edge distance to any gold class, incorrect uniquely resolved predictions only; missing distances excluded and counted',
        relations='Exact > broader > narrower > sibling > other; broader/narrower use directed paths to any gold class',
        consistency='All reachable ancestors, including roots, must be predicted; denominator resolved graph nodes; no resolved nodes => undefined',
        uncertainty='Paired entity bootstrap, 2000 resamples, seed 230917; exploratory, no multiplicity correction',
        caveat='Oracle graph context, previously exposed entities and incomplete gold annotations; out_of_taxonomy is not a hallucination verdict',
        manual_review='Stratified exploratory sample; not an unbiased overall error estimate; existing annotation file preserved')
    (dest / 'interpretation.json').write_text(json.dumps(definitions, indent=2))
    print(f'Scores written to {dest}', flush=True)

if __name__ == '__main__':
    main()
