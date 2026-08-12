"""
Build the public, content-free aggregate artifacts for this repository.

Inputs (kept OUT of the repo because they contain private conversation text):
  * <EVAL_BASE>/comparison.json         - RAGAS-vs-shortcut correlations
  * <EVAL_BASE>/shortcut_scores.json    - shortcut meta/aggregates

Outputs (written into this repo):
  * results/aggregates.csv              - per-metric RAGAS vs shortcut moments
  * results/correlations.csv            - Pearson/Spearman + verdict per metric
  * results/SUMMARY.md                  - one-page plain-English summary
  * figures/correlations.png            - Spearman bar chart with verdict bands

No per-sample rows and no user content are ever written.
"""
import csv
import json
import os
import shutil

from . import config

VERDICT = lambda rho: "STRONG" if abs(rho) >= 0.7 else ("MODERATE" if abs(rho) >= 0.4 else "WEAK")

METRIC_LABELS = {
    "faithfulness": "Faithfulness (NLI)",
    "answer_relevancy": "AnswerRelevancy (TAS-B)",
    "answer_relevancy_nomic": "AnswerRelevancy (nomic)",
    "context_precision": "ContextPrecision (bge)",
    "context_precision_reranked": "ContextPrecision (bge reranked)",
    "context_precision_minilm": "ContextPrecision (MiniLM)",
    "answer_correctness": "AnswerCorrectness (combined)",
    "answer_correctness_rouge": "AnswerCorrectness (ROUGE)",
    "answer_correctness_sem": "AnswerCorrectness (nomic)",
}


def load_comparison():
    with open(config.COMPARISON_JSON) as f:
        return json.load(f)


def load_shortcut_aggregates():
    with open(config.SHORTCUT_JSON) as f:
        return json.load(f)["aggregates"]


def main():
    cmp = load_comparison()
    metrics = cmp["metrics"]
    n_samples = cmp["n_samples"]
    os.makedirs(os.path.dirname(config.AGGREGATES_CSV), exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # --- aggregates.csv -----------------------------------------------------
    with open(config.AGGREGATES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "label", "ragas_mean", "ragas_median", "ragas_std",
                    "shortcut_mean", "shortcut_median", "shortcut_std", "mean_abs_delta", "verdict"])
        for name, m in metrics.items():
            w.writerow([name, METRIC_LABELS.get(name, name),
                        f"{m['ragas_mean']:.4f}", f"{m['ragas_median']:.4f}", f"{m['ragas_std']:.4f}",
                        f"{m['shortcut_mean']:.4f}", f"{m['shortcut_median']:.4f}", f"{m['shortcut_std']:.4f}",
                        f"{m['mean_abs_delta']:.4f}", VERDICT(m['spearman'])])

    # --- correlations.csv ---------------------------------------------------
    with open(config.CORRELATIONS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "label", "pearson", "pearson_p", "spearman", "spearman_p", "verdict"])
        for name, m in metrics.items():
            w.writerow([name, METRIC_LABELS.get(name, name),
                        f"{m['pearson']:.4f}", f"{m['pearson_p']:.4g}",
                        f"{m['spearman']:.4f}", f"{m['spearman_p']:.4g}", VERDICT(m['spearman'])])

    # --- SUMMARY.md ---------------------------------------------------------
    lines = ["# Summary",
             "",
             f"Evaluation of an LLM-agent memory / retrieval shortcut suite against RAGAS LLM-judge scores "
             f"on {n_samples} sampled query/response pairs from a personal opencode session store.",
             "",
             "| Metric | RAGAS mean | shortcut mean | Spearman | verdict |",
             "|---|---|---|---|---|"]
    for name, m in metrics.items():
        lines.append(f"| {METRIC_LABELS.get(name, name)} | {m['ragas_mean']:.3f} | {m['shortcut_mean']:.3f} "
                     f"| {m['spearman']:+.3f} | {VERDICT(m['spearman'])} |")
    lines += [
        "",
        "Headline findings:",
        "1. **AnswerCorrectness** correlates strongly with a cheap ROUGE-L + embedding blend (rho=0.89) — "
        "the heavy judge is replaceable for ranking purposes.",
        "2. **AnswerRelevancy** is well-predicted by nomic query/document cosine (rho=0.71, STRONG); TAS-B is moderate.",
        "3. **Faithfulness** agreement is weak and negatively signed (rho=-0.18): the RAGAS judge saturates at 1.0 "
        "(median 1.000) while the NLI shortcut reads low (median 0.000) — see METHODOLOGY for the oracle-audit caveat.",
        "4. **ContextPrecision** shows no discriminative agreement (rho~0): all variants are near-constant on this "
        "corpus, so the metric cannot rank systems here.",
        "",
        "See `comparison_report.md` in the source eval workspace and `docs/opencode_memory_architecture.md` for "
        "the architectural context.",
        "",
    ]
    with open(config.SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- figures ------------------------------------------------------------
    # copy the source scatter figure if present (numeric, content-free)
    src_fig = os.path.join(config.RESULTS_DIR, "comparison_metrics.png")
    if os.path.exists(src_fig):
        shutil.copy(src_fig, os.path.join(config.FIGURES_DIR, "comparison_metrics.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [METRIC_LABELS.get(k, k) for k in metrics]
    rhos = [metrics[k]["spearman"] for k in metrics]
    colors = ["#2e7d32" if abs(r) >= 0.7 else ("#f9a825" if abs(r) >= 0.4 else "#c62828") for r in rhos]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, rhos, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    for band, label in [(0.4, "MODERATE"), (0.7, "STRONG")]:
        ax.axvline(band, color="gray", ls=":", lw=0.8)
        ax.axvline(-band, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Spearman rho (RAGAS vs shortcut)")
    ax.set_title(f"Agreement between RAGAS judges and local shortcuts (n={n_samples})")
    ax.set_xlim(-1, 1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, "correlations.png"), dpi=150)
    plt.close(fig)

    print(f"wrote {config.AGGREGATES_CSV}")
    print(f"wrote {config.CORRELATIONS_CSV}")
    print(f"wrote {config.SUMMARY_MD}")
    print(f"wrote {os.path.join(config.FIGURES_DIR, 'correlations.png')}")


if __name__ == "__main__":
    main()
