
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORE_PAIRS = [
    ("standard", "graph_aware_standard", "Standard"),
    ("cot_no_graph", "cot_graph", "Chain of thought"),
    ("contrastive_cot_no_graph", "contrastive_cot_graph", "Contrastive CoT"),
]
COMPONENTS = [
    ("standard", "No taxonomy"),
    ("standard_parent_only", "Parent only"),
    ("standard_siblings_only", "Sibling only"),
    ("standard_children_only", "Children only"),
    ("graph_aware_standard", "Full taxonomy"),
]
ALL_CONDITIONS = [
    "standard", "graph_aware_standard", "cot_no_graph", "cot_graph",
    "contrastive_cot_no_graph", "contrastive_cot_graph",
    "standard_parent_only", "standard_siblings_only", "standard_children_only",
]
DISPLAY = {
    "standard": "Standard",
    "graph_aware_standard": "Standard + taxonomy",
    "cot_no_graph": "CoT",
    "cot_graph": "CoT + taxonomy",
    "contrastive_cot_no_graph": "Contrastive CoT",
    "contrastive_cot_graph": "Contrastive CoT + taxonomy",
    "standard_parent_only": "Parent only",
    "standard_siblings_only": "Sibling only",
    "standard_children_only": "Children only",
    "graph_aware_standard_no_children": "Standard taxonomy, children removed",
    "cot_graph_no_children": "CoT taxonomy, children removed",
    "contrastive_cot_graph_no_children": "Contrastive taxonomy, children removed",
}
CHILD_PAIRS = [
    ("graph_aware_standard", "graph_aware_standard_no_children", "Standard"),
    ("cot_graph", "cot_graph_no_children", "Chain of thought"),
    ("contrastive_cot_graph", "contrastive_cot_graph_no_children", "Contrastive CoT"),
]
SEED = 20261009


def ci_mean(x, n=2000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    means = np.empty(n)
    for i in range(n):
        means[i] = x[rng.integers(0, len(x), len(x))].mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_scores(results: Path) -> pd.DataFrame:
    path = results / "analysis" / "entity_scores.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/score_gemini.py for this result folder first.")
    d = pd.read_csv(path)
    if d.duplicated(["entity_id", "condition"]).any():
        raise SystemExit(f"Duplicate entity/condition rows in {path}")
    return d


def load_ranks(results: Path) -> pd.DataFrame:
    path = results / "analysis" / "gold_type_ranks.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the scorer first.")
    return pd.read_csv(path)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def graph_pair_figure(d: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, name: str):
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
    for ax, (base, graph, family) in zip(axs, CORE_PAIRS):
        wide = d[d.condition.isin([base, graph])].pivot(index="entity_id", columns="condition", values=metric).dropna()
        means = [wide[base].mean(), wide[graph].mean()]
        delta = (wide[graph] - wide[base]).to_numpy()
        lo, hi = ci_mean(delta)
        ax.plot([0, 1], means, marker="o", linewidth=2.2)
        ax.set_xticks([0, 1], ["No taxonomy", "Taxonomy"])
        ax.set_title(family)
        ax.grid(axis="y", alpha=.2)
        ax.text(.5, max(means) + .04, f"paired Δ = {delta.mean():+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",
                ha="center", va="bottom", fontsize=9)
        ax.set_xlim(-.25, 1.25)
    axs[0].set_ylabel(ylabel)
    if metric != "types_count":
        axs[0].set_ylim(0, 1)
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    save(fig, out, name)


def component_figure(d: pd.DataFrame, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, metric, title in [
        (axs[0], "gold_coverage", "Gold coverage"),
        (axs[1], "ndcg", "Depth-weighted nDCG"),
    ]:
        means, lows, highs, labels = [], [], [], []
        for cond, label in COMPONENTS:
            x = d.loc[d.condition == cond, metric].to_numpy(float)
            lo, hi = ci_mean(x)
            means.append(x.mean()); lows.append(lo); highs.append(hi); labels.append(label)
        y = np.arange(len(labels))
        ax.errorbar(means, y,
                    xerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
                    fmt="o", capsize=3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="x", alpha=.2); ax.set_title(title)
        ax.set_xlabel("Mean across the same entities (95% entity-bootstrap CI)")
    fig.suptitle("Which taxonomy component helps Standard prompting?", fontsize=14, weight="bold")
    fig.tight_layout()
    save(fig, out, "03_component_ablation")


def rank_profile(d: pd.DataFrame, r: pd.DataFrame, out: Path):
    # MRR is based only on the first match among deepest gold labels. Make those discrete
    # outcomes visible directly instead of drawing a threshold/survival curve.
    max_depth = r.groupby("entity_id").gold_depth.max()
    rr = r.copy()
    rr["max_depth"] = rr.entity_id.map(max_depth)
    rr = rr[rr.gold_depth == rr.max_depth]
    first = rr.groupby(["entity_id", "condition"])["rank"].min().reset_index()
    def cat(v):
        if pd.isna(v): return "Miss"
        if v == 1: return "Rank 1"
        if v == 2: return "Rank 2"
        return "Rank 3+"
    first["category"] = first["rank"].map(cat)
    order = [c for pair in CORE_PAIRS for c in pair[:2]]
    cats = ["Rank 1", "Rank 2", "Rank 3+", "Miss"]
    tab = (pd.crosstab(first.condition, first.category, normalize="index")
             .reindex(index=order, columns=cats, fill_value=0) * 100)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    left = np.zeros(len(tab))
    for cat_name in cats:
        vals = tab[cat_name].to_numpy()
        ax.barh(np.arange(len(tab)), vals, left=left, label=cat_name)
        left += vals
    ax.set_yticks(np.arange(len(tab)), [DISPLAY[c] for c in tab.index])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Entities (%)")
    ax.set_title("Where does the first deepest-gold match appear?", weight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.10))
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    save(fig, out, "04_most_specific_rank_profile")
    tab.to_csv(out.parent / "rank_profile_percent.csv")


def target_depths(results: Path) -> pd.Series:
    values = {}
    # Read one result per entity. The runner/scorer embeds the same metadata in every condition.
    for p in sorted(results.glob("example_*/*_result.json")):
        x = json.loads(p.read_text(encoding="utf-8"))
        eid = x.get("entity_id")
        if eid in values or x.get("status") != "completed":
            continue
        meta = x.get("metadata", {})
        ref = meta.get("reference_class_id")
        depths = [g.get("depth") for g in meta.get("gold_types", []) if g.get("class_id") == ref]
        if depths:
            values[eid] = int(depths[0])
    if not values:
        raise SystemExit("Could not recover oracle-centre depths from result metadata")
    return pd.Series(values, name="target_depth")


def depth_uplift(d: pd.DataFrame, results: Path, out: Path):
    depths = target_depths(results)
    # Bins chosen from the actual 1k support: 4-5, 6-7, 8-11.
    bins = [(4, 5, "Depth 4–5"), (6, 7, "Depth 6–7"), (8, 99, "Depth 8+")]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    records = []
    xloc = np.arange(len(bins))
    for j, (base, graph, family) in enumerate(CORE_PAIRS):
        wide = d[d.condition.isin([base, graph])].pivot(index="entity_id", columns="condition", values="gold_coverage").dropna()
        delta = wide[graph] - wide[base]
        means = []; lows = []; highs = []
        for lo_d, hi_d, label in bins:
            ids = depths[(depths >= lo_d) & (depths <= hi_d)].index.intersection(delta.index)
            x = delta.loc[ids].to_numpy(float)
            lo, hi = ci_mean(x); m = x.mean()
            means.append(m); lows.append(lo); highs.append(hi)
            records.append({"family": family, "depth_bin": label, "n": len(x), "mean_delta_gold_coverage": m, "ci_low": lo, "ci_high": hi})
        offset = (j - 1) * .16
        ax.errorbar(xloc + offset, means,
                    yerr=[np.array(means)-np.array(lows), np.array(highs)-np.array(means)],
                    marker="o", linestyle="none", capsize=3, label=family)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(xloc, [b[2] for b in bins])
    ax.set_ylabel("Paired change in gold coverage\n(taxonomy − no taxonomy)")
    ax.set_title("Does taxonomy help differently at different target depths?", weight="bold")
    ax.legend(); ax.grid(axis="y", alpha=.18)
    fig.tight_layout(); save(fig, out, "05_graph_uplift_by_target_depth")
    pd.DataFrame(records).to_csv(out.parent / "graph_uplift_by_target_depth.csv", index=False)


def model_comparison(primary: pd.DataFrame, other: pd.DataFrame, labels: tuple[str, str], out: Path):
    # Same 1k entity IDs and same nine conditions are required for a fair paired cohort comparison.
    if set(primary.entity_id) != set(other.entity_id):
        raise SystemExit("Model comparison requires identical entity cohorts")
    conds = [c for c in ALL_CONDITIONS if c in set(primary.condition) and c in set(other.condition)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(conds))
    offsets = [-.11, .11]
    for df, label, offset in [(primary, labels[0], offsets[0]), (other, labels[1], offsets[1])]:
        means=[]; lows=[]; highs=[]
        for c in conds:
            x=df.loc[df.condition==c,'gold_coverage'].to_numpy(float);lo,hi=ci_mean(x)
            means.append(x.mean());lows.append(lo);highs.append(hi)
        ax.errorbar(means, y+offset,
                    xerr=[np.array(means)-np.array(lows),np.array(highs)-np.array(means)],
                    fmt='o',capsize=3,label=label)
    ax.set_yticks(y,[DISPLAY[c] for c in conds]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Mean gold coverage (95% entity-bootstrap CI)')
    ax.set_title('Model comparison on the same 1,000 entities',weight='bold');ax.legend();ax.grid(axis='x',alpha=.18)
    fig.tight_layout();save(fig,out,'06_model_comparison_gold_coverage')

    fig,ax=plt.subplots(figsize=(9.5,5.5)); records=[]
    x=np.arange(3)
    for df,label,offset in [(primary,labels[0],-.1),(other,labels[1],.1)]:
        means=[];lows=[];highs=[]
        for base,graph,family in CORE_PAIRS:
            w=df[df.condition.isin([base,graph])].pivot(index='entity_id',columns='condition',values='gold_coverage').dropna()
            delta=(w[graph]-w[base]).to_numpy();lo,hi=ci_mean(delta);m=delta.mean()
            means.append(m);lows.append(lo);highs.append(hi);records.append({'model':label,'family':family,'n':len(delta),'mean_delta':m,'ci_low':lo,'ci_high':hi})
        ax.errorbar(x+offset,means,yerr=[np.array(means)-np.array(lows),np.array(highs)-np.array(means)],fmt='o',capsize=3,label=label)
    ax.axhline(0,linewidth=1);ax.set_xticks(x,[p[2] for p in CORE_PAIRS]);ax.set_ylabel('Graph uplift in gold coverage')
    ax.set_title('Does taxonomy context help both models?',weight='bold');ax.legend();ax.grid(axis='y',alpha=.18)
    fig.tight_layout();save(fig,out,'07_model_comparison_graph_uplift')
    pd.DataFrame(records).to_csv(out.parent/'model_graph_uplift.csv',index=False)


def children_comparison(main: pd.DataFrame, child: pd.DataFrame, out: Path):
    # The no-child run contains only child-present entities. Restrict the WITH-child
    # side to exactly those IDs so this is a matched comparison.
    ids=set(child.entity_id)
    if not ids:
        raise SystemExit('No child-ablation entity scores found')
    records=[]
    fig,axs=plt.subplots(1,2,figsize=(13.5,5),sharey=False)
    for ax,metric,title in [(axs[0],'gold_coverage','Gold coverage'),(axs[1],'ndcg','Depth-weighted nDCG')]:
        for i,(with_c,without_c,family) in enumerate(CHILD_PAIRS):
            a=main[(main.condition==with_c)&main.entity_id.isin(ids)].set_index('entity_id')[metric]
            b=child[(child.condition==without_c)&child.entity_id.isin(ids)].set_index('entity_id')[metric]
            common=a.index.intersection(b.index);a=a.loc[common];b=b.loc[common]
            if len(common)!=len(ids):
                raise SystemExit(f'Children comparison missing paired results for {family}: {len(common)}/{len(ids)}')
            means=[b.mean(),a.mean()];delta=(a-b).to_numpy();lo,hi=ci_mean(delta)
            ax.plot([0,1],means,marker='o',linewidth=2,label=family)
            records.append({'metric':metric,'family':family,'n':len(common),'without_children_mean':b.mean(),'with_children_mean':a.mean(),'paired_delta_with_minus_without':delta.mean(),'ci_low':lo,'ci_high':hi})
        ax.set_xticks([0,1],['Children removed','Children included']);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel(f'Mean score on the same child-present entities (n={len(ids)})')
    axs[1].legend(loc='best')
    fig.suptitle('What changes when direct-child evidence is removed?',weight='bold',fontsize=14)
    fig.tight_layout();save(fig,out,'08_children_with_vs_without')
    pd.DataFrame(records).to_csv(out.parent/'children_with_vs_without.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',required=True,help='Primary scored result directory')
    ap.add_argument('--primary-label',default='Primary model')
    ap.add_argument('--compare-results',default=None)
    ap.add_argument('--compare-label',default='Comparison model')
    ap.add_argument('--children-results',default=None,help='Scored no-children result directory for the primary model')
    args=ap.parse_args()

    results=Path(args.results).resolve();d=load_scores(results);r=load_ranks(results)
    needed=set(c for pair in CORE_PAIRS for c in pair[:2])
    if not needed.issubset(set(d.condition)):
        raise SystemExit('Primary result folder is missing one or more core graph/no-graph conditions')
    out=results/'analysis'/'final_figures'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

    graph_pair_figure(d,'gold_coverage','Mean gold coverage','Does taxonomy context improve gold-type coverage?',out,'01_graph_vs_no_graph_gold_coverage')
    graph_pair_figure(d,'ndcg','Mean depth-weighted nDCG','Does taxonomy context improve ranked gold-type recovery?',out,'02_graph_vs_no_graph_ndcg')
    if all(c in set(d.condition) for c,_ in COMPONENTS):
        component_figure(d,out)
    rank_profile(d,r,out)
    depth_uplift(d,results,out)

    if args.compare_results:
        other=load_scores(Path(args.compare_results).resolve())
        model_comparison(d,other,(args.primary_label,args.compare_label),out)
    if args.children_results:
        child=load_scores(Path(args.children_results).resolve())
        children_comparison(d,child,out)

    manifest={
        'primary_results':str(results),
        'primary_label':args.primary_label,
        'compare_results':args.compare_results,
        'compare_label':args.compare_label if args.compare_results else None,
        'children_results':args.children_results,
        'removed_from_previous_suite':[
            'MRR threshold/survival plot: MRR is a small discrete reciprocal-rank variable; direct rank categories are clearer.',
            'Gold-pair relative-depth recall plot: it used max-gold-depth minus label depth, not verified ancestor steps, and pair-weighting created very uneven support.'
        ],
    }
    (results/'analysis'/'final_plot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Final figures:',out)

if __name__=='__main__':
    main()
