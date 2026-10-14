"""Historical metric semantics, extracted without importing archive code.

Graph edges point child to parent; ranking retains original output positions.
"""
import math
import csv
import re
from collections import defaultdict
from functools import lru_cache
import numpy as np
import networkx as nx

def normalize_label(text):
    text = text.casefold().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")

def canonical_prediction(label, aliases):
    normalized = normalize_label(label)
    return aliases.get(normalized, normalized)

def relevance_lookup(gold_types):
    """
    Assign hierarchical relevance from WiKC depth.

    The deepest listed type receives relevance 1.0. Broader listed
    classes receive partial credit proportional to their depth.
    """
    maximum_depth = max(item["depth"] for item in gold_types)
    lookup = {}

    for item in gold_types:
        label = normalize_label(item["label"])
        relevance = (
            item["depth"] / maximum_depth
            if maximum_depth > 0
            else 1.0
        )
        lookup[label] = max(lookup.get(label, 0.0), relevance)

    most_specific_labels = {
        normalize_label(item["label"])
        for item in gold_types
        if item["depth"] == maximum_depth
    }

    return lookup, most_specific_labels

def dcg_at_k(relevances, k):
    return sum(
        (2.0 ** relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )

def ndcg(predictions, gold_types, aliases, k=None):
    gold_relevance, _ = relevance_lookup(gold_types)

    if k is None:
        k = max(len(predictions), len(gold_relevance), 1)

    observed_relevances = []
    matched_gold_labels = set()

    for prediction in predictions[:k]:
        canonical = canonical_prediction(prediction, aliases)
        relevance = gold_relevance.get(canonical, 0.0)

        if canonical in matched_gold_labels:
            relevance = 0.0
        elif relevance > 0:
            matched_gold_labels.add(canonical)

        observed_relevances.append(relevance)

    while len(observed_relevances) < k:
        observed_relevances.append(0.0)

    ideal_relevances = sorted(
        gold_relevance.values(),
        reverse=True,
    )[:k]

    while len(ideal_relevances) < k:
        ideal_relevances.append(0.0)

    dcg = dcg_at_k(observed_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

def reciprocal_rank(predictions, gold_types, aliases, k=None):
    _, most_specific_labels = relevance_lookup(gold_types)
    inspected = predictions if k is None else predictions[:k]

    for rank, prediction in enumerate(inspected, start=1):
        if canonical_prediction(prediction, aliases) in most_specific_labels:
            return 1.0 / rank

    return 0.0

def hit_at_k(predictions, gold_types, aliases, k=None):
    _, most_specific_labels = relevance_lookup(gold_types)
    inspected = predictions if k is None else predictions[:k]

    return int(
        any(
            canonical_prediction(prediction, aliases)
            in most_specific_labels
            for prediction in inspected
        )
    )

def build_graph(file_path):
    graph = nx.DiGraph()

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) == 2:
                child, parent = row
                graph.add_edge(child, parent)

    return graph

def load_label_index(file_path, wanted=None):
    """Map normalized WiKC labels to one or more class IDs."""
    label_to_ids = defaultdict(set)
    id_to_label = {}

    pattern = re.compile(
        r'^(wd:Q\d+)\s+rdfs:label\s+"((?:\\.|[^"])*)"\s+\.$'
    )

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line.strip())

            if not match:
                continue

            class_id = match.group(1)
            label = match.group(2).replace(r'\"', '"')
            normalized = normalize_label(label)

            if wanted is not None and normalized not in wanted:
                continue
            label_to_ids[normalized].add(class_id)
            id_to_label[class_id] = label

    return dict(label_to_ids), id_to_label

def set_metrics(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * tp / (len(pred) + len(gold)) if pred or gold else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp,
                predicted_count=len(pred), gold_count=len(gold), strict_accuracy=int(pred == gold))

class Taxonomy:
    def __init__(self, graph, labels):
        self.graph = graph
        self.labels = labels
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
        norm = normalize_label
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
