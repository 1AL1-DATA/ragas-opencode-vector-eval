"""
Compare RAGAS (LLM-judge) results against the local shortcut scores.

Loads the latest results/results-<ts>.json and shortcut_scores.json, joins on
sample order (both artifacts key samples identically), and for every metric
reports Pearson + Spearman correlation, RAGAS vs shortcut moments, and a
scatter + rank-rank figure per metric.

Output: comparison.json (aggregates only), comparison_report.md, and
results/comparison_metrics.png. Per-sample rows and conversation content are
deliberately NOT written (the source artifacts remain in the private eval
workspace).

NOTE: this script requires the private eval artifacts (dataset-derived
`results-*.json` / `shortcut_scores.json`) and is intended to be run from the
original workspace via EVAL_BASE, not from this repository.
"""
import json
import os
import re

import numpy as np

from . import config

METRICS = [
    ("faithfulness", "faithfulness", ("faithfulness", "score"), "sentence-level NLI (DeBERTa-v3 MNLI)"),
    ("answer_relevancy", "answer_relevancy", ("answer_relevancy", "tas_b_dot"), "TAS-B dot(q, a)"),
    ("answer_relevancy_nomic", "answer_relevancy", ("answer_relevancy", "nomic_query_doc_cosine"), "nomic query/doc cosine"),
    ("context_precision", "context_precision", ("context_precision", "bge_reranker_v2_m3", "retrieved_order"), "bge-reranker-v2-m3 NDCG@5"),
    ("context_precision_reranked", "context_precision", ("context_precision", "bge_reranker_v2_m3", "reranked_order"), "bge-reranker-v2-m3 NDCG@5 (reranked)"),
    ("context_precision_minilm", "context_precision", ("context_precision", "ms_marco_minilm", "retrieved_order"), "ms-marco-MiniLM NDCG@5"),
    ("answer_correctness", "answer_correctness", ("answer_correctness", "combined"), "0.75*ROUGE-L + 0.25*nomic cosine"),
    ("answer_correctness_rouge", "answer_correctness", ("answer_correctness", "rouge_l_f1"), "ROUGE-L F1 only"),
    ("answer_correctness_sem", "answer_correctness", ("answer_correctness", "nomic_cosine"), "nomic cosine only"),
]

_MODEL_KEYS = {
    "bge_reranker_v2_m3": "BAAI/bge-reranker-v2-m3",
    "ms_marco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def latest_results():
    if not os.path.isdir(config.RESULTS_DIR):
        return None
    files = [f for f in os.listdir(config.RESULTS_DIR) if re.match(r"results-\d{8}-\d{6}\.json", f)]
    if not files:
        return None
    return os.path.join(config.RESULTS_DIR, sorted(files)[-1])


def get(d, path):
    for k in path:
        if isinstance(d, dict) and k in _MODEL_KEYS and k not in d and _MODEL_KEYS[k] in d:
            k = _MODEL_KEYS[k]
        d = d[k]
    return d


def main():
    res_path = latest_results()
    if not res_path:
        raise SystemExit(f"no results/results-*.json found ({config.RESULTS_DIR})")
    if not os.path.exists(config.SHORTCUT_JSON):
        raise SystemExit(f"missing {config.SHORTCUT_JSON} — run shortcut_metrics.py first")

    with open(res_path) as f:
        res = json.load(f)
    with open(config.SHORTCUT_JSON) as f:
        short = json.load(f)

    ragas = {d["user_input"]: d for d in res["per_sample"]}
    short_rows = {d["sample_id"]: d for d in short["per_sample"]}
    # join by order, not by content: both artifacts list the same dataset rows.
    ragas_by_idx = {i: d for i, d in enumerate(res["per_sample"])}
    common = [i for i in range(min(len(ragas_by_idx), len(short_rows)))
              if ragas_by_idx[i].get("user_input") is not None]
    print(f"joined {len(common)}/{len(ragas_by_idx)} samples (RAGAS {res_path})\n")

    import scipy.stats as st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = []
    comparison = {"source_results": os.path.basename(res_path), "n_samples": len(common), "metrics": {}}
    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.5 * len(METRICS), 8))
    if len(METRICS) == 1:
        axes = axes.reshape(2, 1)

    for mi, (name, col, path, label) in enumerate(METRICS):
        pairs = []
        for i in common:
            rv = ragas_by_idx[i].get(col)
            sv = get(short_rows[i], path)
            if rv is None or sv is None or not np.isfinite(float(rv)):
                continue
            pairs.append((float(rv), float(sv)))
        if len(pairs) < 3:
            print(f"  {name}: only {len(pairs)} valid pairs, skipping")
            continue
        r = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])

        pear = st.pearsonr(r, s)
        spear = st.spearmanr(r, s)
        delta = s - r
        rows = {
            "ragas_mean": float(np.mean(r)), "ragas_median": float(np.median(r)), "ragas_std": float(np.std(r)),
            "shortcut_mean": float(np.mean(s)), "shortcut_median": float(np.median(s)), "shortcut_std": float(np.std(s)),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "pearson": float(pear.statistic), "pearson_p": float(pear.pvalue),
            "spearman": float(spear.statistic), "spearman_p": float(spear.pvalue),
        }
        comparison["metrics"][name] = rows

        ax1, ax2 = axes[0, mi], axes[1, mi]
        ax1.scatter(r, s, s=12, alpha=0.5)
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        ax1.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ax1.set_xlabel("RAGAS"); ax1.set_ylabel("shortcut"); ax1.set_title(f"{name}  rho={spear.statistic:.2f}")
        ax2.scatter(st.rankdata(r), st.rankdata(s), s=12, alpha=0.5)
        ax2.set_xlabel("RAGAS rank"); ax2.set_ylabel("shortcut rank"); ax2.set_title("rank-rank")

        verdict = "STRONG" if abs(spear.statistic) >= 0.7 else ("MODERATE" if abs(spear.statistic) >= 0.4 else "WEAK")
        report.append(
            f"\n### {name} — {label}\n\n"
            f"| | RAGAS | shortcut |\n|---|---|---|\n"
            f"| mean | {rows['ragas_mean']:.3f} | {rows['shortcut_mean']:.3f} |\n"
            f"| median | {rows['ragas_median']:.3f} | {rows['shortcut_median']:.3f} |\n"
            f"| std | {rows['ragas_std']:.3f} | {rows['shortcut_std']:.3f} |\n\n"
            f"- **Pearson** = {rows['pearson']:.3f} (p={rows['pearson_p']:.3g})  "
            f"**Spearman** = {rows['spearman']:.3f} (p={rows['spearman_p']:.3g}) — agreement: **{verdict}**\n"
            f"- mean |delta| = {rows['mean_abs_delta']:.3f} "
            f"(shortcut reads {'higher' if np.mean(delta) > 0 else 'lower'} on average)\n\n"
        )
        print(f"  {name:30s} pearson={rows['pearson']:+.3f} spearman={rows['spearman']:+.3f}  ({verdict})")

    fig.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for _ in range(len(METRICS)):
        for _ in range(2):
            fig.savefig(os.path.join(config.RESULTS_DIR, "comparison_metrics.png"), dpi=100)
    plt.close(fig)

    with open(config.COMPARISON_JSON, "w") as f:
        json.dump(comparison, f, indent=2)

    header = (
        f"# Shortcut vs RAGAS comparison\n\n"
        f"Results: `{os.path.basename(res_path)}` — {len(common)} joined samples.\n\n"
        "Verdict scale: |Spearman| >= 0.7 STRONG, 0.4-0.7 MODERATE, < 0.4 WEAK.\n"
        "Aggregate numbers only; per-sample rows are excluded from this report.\n"
        f"Figures: `results/comparison_metrics.png`.\n"
    )
    with open(os.path.join(config.BASE, "comparison_report.md"), "w") as f:
        f.write(header + "\n".join(report) + "\n")
    print("\nwrote comparison.json and comparison_report.md")


if __name__ == "__main__":
    main()
