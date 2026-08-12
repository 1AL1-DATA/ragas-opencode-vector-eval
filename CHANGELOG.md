# Changelog

All notable changes to this repository are documented here, following
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [0.1.0] - 2026-08-12

Initial release.

### Added
- RAGAS-vs-shortcut agreement study on a real agentic-RAG vector store.
- Deterministic shortcut suite (`src/shortcut_metrics.py`): graded NDCG
  context precision (bge / MiniLM, retrieved & reranked), sentence-level NLI
  faithfulness (DeBERTa-v3 MNLI), TAS-B + nomic AnswerRelevancy, ROUGE-L +
  nomic AnswerCorrectness.
- Agreement analysis (`src/compare_eval.py`): Pearson/Spearman per metric,
  scatter + rank-rank figures.
- Aggregate artifacts (`src/build_aggregates.py`): `results/aggregates.csv`,
  `results/correlations.csv`, `results/SUMMARY.md`, `figures/correlations.png`.
- Narrative layer: `README.md`, `research_report.md`, `blog_post.md`,
  `linkedin_post.md`, `lit_digest.md`, `arxiv_paper.tex`.
- `docs/opencode_memory_architecture.md` — how the vector store + plugin
  manage opencode agent memory, and how this study's system was derived.
- Data policy (`data/README.md`): no conversation content is shipped;
  per-sample rows excluded by design.
- Unit tests for pure shortcut functions (`tests/`).

### Findings (2026-08-10/12, 99 joined samples)
- AnswerCorrectness vs. ROUGE-L + cosine: Spearman 0.894 (STRONG).
- AnswerRelevancy vs. nomic query/doc cosine: 0.709 (STRONG);
  TAS-B: 0.576 (MODERATE).
- Faithfulness vs. sentence NLI: −0.185 (WEAK; judge median 1.000 vs NLI 0.000).
- ContextPrecision vs. graded NDCG: |ρ| < 0.13 (WEAK; corpus-degenerate).
