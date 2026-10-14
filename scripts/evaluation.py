"""Flat set and ancestor-augmented hierarchy metrics; edges point child -> parent.

This implements hierarchical F1, NOT the LCA-F1 algorithm. Ambiguous labels are
unresolved unless an exact gold label identifies their class(es). Unresolved
predictions remain opaque false-positive nodes in hierarchical precision.
"""
from functools import lru_cache
import numpy as np
import networkx as nx
from common import ORIGINAL
import score_ranked_results as legacy

SEED = 230917
PAIRS = [('standard', 'graph_aware_standard'), ('cot_no_graph', 'cot_graph'),
         ('contrastive_cot_no_graph', 'contrastive_cot_graph')]

def set_metrics(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * tp / (len(pred) + len(gold)) if pred or gold else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp,
                predicted_count=len(pred), gold_count=len(gold), strict_accuracy=int(pred == gold))

def paired_summary(values, resamples=2000):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Paired comparison requires finite, nonempty observations')
    rng = np.random.default_rng(SEED)
    boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(resamples)]
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(n=len(x), mean_delta=x.mean(), ci_low=lo, ci_high=hi,
                helped=int((x > 1e-12).sum()), tied=int((abs(x) <= 1e-12).sum()),
                harmed=int((x < -1e-12).sum()))

class Taxonomy:
    def __init__(self, graph=None, labels=None):
        self.graph = graph if graph is not None else legacy.build_graph(ORIGINAL / 'data/wikc.tsv')
        self.labels = labels if labels is not None else legacy.load_label_index(ORIGINAL / 'data/wiki_taxonomy_labels.tsv')[0]
        self.undirected = self.graph.to_undirected(as_view=True)
        self.is_dag = nx.is_directed_acyclic_graph(self.graph)

    @lru_cache(maxsize=None)
    def ancestors(self, node):
        return frozenset(nx.descendants(self.graph, node)) if node in self.graph else frozenset()

    @lru_cache(maxsize=8)
    def distances(self, gold_ids):
        sources = [g for g in gold_ids if g in self.graph]
        if not sources:
            return {}, {}, {}
        # Undirected shortest path; upward from gold => broader predictions.
        def bfs(graph):
            return nx.multi_source_dijkstra_path_length(graph, sources, weight=None)
        return bfs(self.undirected), bfs(self.graph), bfs(self.graph.reverse(copy=False))

    def score(self, predictions, gold):
        norm = legacy.normalize_label
        pred = list(dict.fromkeys(norm(x) for x in predictions))
        gold_by_label = {}
        for g in gold:
            gold_by_label.setdefault(norm(g['label']), set()).add(g['class_id'])
        flat = set_metrics(pred, gold_by_label)
        gold_ids = set().union(*gold_by_label.values())
        gold_augmented = gold_ids | set().union(*(self.ancestors(g) for g in gold_ids))
        mapped, rows = set(), []
        distances, broader, narrower = self.distances(tuple(sorted(gold_ids)))
        parents_gold = set().union(*(set(self.graph.successors(g)) for g in gold_ids if g in self.graph))
        for label in pred:
            candidates = self.labels.get(label, set())
            exact = label in gold_by_label
            ids = gold_by_label[label] if exact else (set(candidates) if len(candidates) == 1 else set())
            mapped.update(ids)
            distance = min((distances[v] for v in ids if v in distances), default=None)
            direction_distance = None
            if exact:
                category, bucket, distance = 'exact_gold', 'Exact gold', 0
            elif not candidates:
                category, bucket = 'out_of_taxonomy', 'Outside WiKC'
            elif not ids:
                category, bucket = 'ambiguous', 'Ambiguous label'
            elif not any(v in self.graph for v in ids):
                category, bucket = 'unconnected_label', 'No graph node'
            elif any(v in broader for v in ids):
                direction_distance = min(broader[v] for v in ids if v in broader)
                category = 'broader'
                bucket = f'Broader {direction_distance}' if direction_distance < 3 else 'Broader 3+'
            elif any(v in narrower for v in ids):
                direction_distance = min(narrower[v] for v in ids if v in narrower)
                category = 'narrower'
                bucket = 'Narrower 1' if direction_distance == 1 else 'Narrower 2+'
            elif any(set(self.graph.successors(v)) & parents_gold for v in ids if v in self.graph):
                category, bucket = 'sibling', 'Sibling'
            else:
                category, bucket = ('other_taxonomy', 'Other WiKC') if distance is not None else ('disconnected', 'Disconnected')
            rows.append(dict(prediction=label, correct=int(exact), category=category,
                error_bucket=bucket, distance=distance, direction_distance=direction_distance,
                candidate_ids='|'.join(sorted(candidates)), resolved_ids='|'.join(sorted(ids))))
        pred_augmented = mapped | set().union(*(self.ancestors(v) for v in mapped))
        # Never let unresolved labels disappear from the precision denominator.
        pred_augmented.update('unresolved:' + r['prediction'] for r in rows if not r['resolved_ids'])
        hierarchy = set_metrics(pred_augmented, gold_augmented)
        graph_nodes = {v for v in mapped if v in self.graph}
        consistent = sum(self.ancestors(v) <= mapped for v in graph_nodes)
        errors = [r['distance'] for r in rows if not r['correct'] and r['distance'] is not None]
        flat.update(hierarchical_precision=hierarchy['precision'], hierarchical_recall=hierarchy['recall'],
            hierarchical_f1=hierarchy['f1'], hierarchy_consistency=consistent / len(graph_nodes) if graph_nodes else np.nan,
            hierarchy_consistent_set=int(consistent == len(graph_nodes)) if graph_nodes else np.nan,
            hierarchy_resolved_nodes=len(graph_nodes), unresolved_predictions=sum(not r['resolved_ids'] for r in rows),
            taxonomy_error_mean=float(np.mean(errors)) if errors else np.nan,
            taxonomy_error_median=float(np.median(errors)) if errors else np.nan,
            taxonomy_error_count=len(errors), target_depth=max(g['depth'] for g in gold))
        return flat, rows
