# Methodology — how the evaluation was built and run

This document is the reproduction guide. It describes every step from the raw
vector store to the aggregate numbers in `results/`. All numbers quoted here
are the published aggregates; per-sample content is intentionally absent.

## 1. System under test

| Component | Details |
|---|---|
| Session store | opencode (SQLite), ~427 sessions / ~22.8k messages at eval time |
| Vector store | Chroma, collection `opencode_sessions`, cosine, `nomic-embed-text` (768-d, GPU via Ollama) |
| Generator | `gemma4:12b` via Ollama (GPU), `num_ctx=4096`, temperature 0 |
| Judge | `gemma4:12b` via Ollama (same model family as generator) |

The vector store is a *derived index* of the session store: `extract.py`
reads the materialized `message`/`part` tables and writes a JSONL corpus;
`embed.py` wipes and re-embeds the Chroma collection. Retrieval is
top-k cosine search over that collection; the generator answers given the
retrieved chunks. See `docs/opencode_memory_architecture.md` for the full
pipeline.

## 2. Dataset construction

Sampled deterministically (`seed=42`, 100 samples, max 5 per session) from the
session store:

- **Query** — a real user message of length 15–1500 chars.
- **Reference** — the historical assistant reply (used as ground truth).
- **Retrieved contexts** — top-5 chunks returned by the vector store for the
  query, truncated per chunk to ~1000 chars.
- **Response** — a *fresh* generation by `gemma4:12b` given the same top-5
  context (temperature 0), capped at 1200 tokens. Rationale: we evaluate the
  *retrieval pipeline* (context → answer), not the model's memory of what it
  once said.

After filtering samples where the judge returned no usable verdict, the RAGAS
and shortcut runs joined on **99** samples. All correlations in this repo are
computed on that joined set.

## 3. RAGAS (LLM-judge) pass

Library: `ragas` 0.4.x, classic `ragas.metrics` classes, fully local
(`LangchainLLMWrapper(ChatOllama(gemma4:12b))`,
`LangchainEmbeddingsWrapper(OllamaEmbeddings(nomic-embed-text))`).

Metrics:

| RAGAS metric | What it judges |
|---|---|
| Faithfulness | is the answer grounded in the retrieved context? |
| AnswerRelevancy | does the answer address the question? |
| ContextPrecision | are relevant chunks ranked near the top? |
| AnswerCorrectness | does the answer match the historical reply? |

Run config: `max_workers=1` (Ollama serves one model request at a time),
`timeout=900s`, `max_retries=3`. The full pass took **8h26m** for 400
samples (4 metrics × 100) — the cost that motivates this study.

RAGAS aggregates on the joined set:

| Metric | mean | median | std |
|---|---|---|---|
| Faithfulness | 0.770 | 1.000 | 0.398 |
| AnswerRelevancy | 0.519 | 0.631 | 0.318 |
| ContextPrecision | 0.633 | 0.700 | 0.383 |
| AnswerCorrectness | 0.330 | 0.222 | 0.249 |

## 4. Shortcut pass (fully deterministic, local)

All shortcuts run on one GPU via PyTorch / sentence-transformers /
Ollama embeddings. Total wall time ≈ 55 min on CPU (torch was built for a
newer CUDA than the driver, forcing CPU fallback).

### ContextPrecision → graded NDCG

The naive "threshold-based precision@k" shortcut collapsed (lexical labels
marked ~89% of chunks relevant → τ ≈ 0), so we use a **graded, threshold-free**
design:

- **Relevance** of chunk *c* to reference *r*: `σ(score(r, c))` via
  cross-encoder (bge-reranker-v2-m3 or ms-marco-MiniLM-L-6-v2).
- **Order** is the query-vs-chunk reranker ranking (retrieved order) or the
  reranked order.
- Metric = NDCG over graded relevance under that order; 1.0 iff relevance is
  perfectly ranked.

Shortcut aggregates:

| variant | retrieved_order NDCG | reranked_order NDCG |
|---|---|---|
| bge-reranker-v2-m3 | 0.990 | 0.991 |
| ms-marco-MiniLM-L-6-v2 | 0.755 | 0.759 |

### Faithfulness → sentence-level NLI

Split the response into sentences (regex), score each `(chunk, sentence)` pair
with DeBERTa-v3 MNLI entailment probability, take the max over chunks, and
count a sentence as grounded if that probability ≥ 0.5.

`score = grounded_sentences / n_sentences` → aggregate mean **0.301** (median
0.000), i.e. the strict view: most sentences are not entailed by the retrieved
chunks.

### AnswerRelevancy → asymmetric QA embeddings

Two independent shortcuts:

- **TAS-B** `msmarco-distilbert-base-tas-b`: dot product of query/response
  embeddings → mean 98.0 (saturating scale).
- **nomic** `nomic-embed-text` with `search_query:` / `search_document:`
  prefixes: cosine(query, response) → mean **0.710**.

### AnswerCorrectness → ROUGE-L + semantic blend

`combined = 0.75·ROUGE-L-F1(response, reference) + 0.25·cosine(nomic(response), nomic(reference))`
→ aggregate mean **0.291** (ROUGE alone 0.146, semantic alone 0.723).

## 5. Agreement analysis

For every metric we compute Pearson and Spearman correlation between the RAGAS
score and the shortcut score over the 99 joined samples, plus per-metric
moments and mean |Δ|. Verdict bands:

| |Spearman| | Verdict |
|---|---|---|
| ≥ 0.7 | STRONG |
| 0.4–0.7 | MODERATE |
| < 0.4 | WEAK |

Results table (`results/correlations.csv`):

| Metric | Pearson | Spearman | Verdict |
|---|---|---|---|
| AnswerCorrectness (combined) | 0.822 | **0.894** | STRONG |
| AnswerCorrectness (ROUGE) | 0.788 | **0.870** | STRONG |
| AnswerCorrectness (nomic) | 0.775 | **0.875** | STRONG |
| AnswerRelevancy (nomic) | 0.777 | **0.709** | STRONG |
| AnswerRelevancy (TAS-B) | 0.556 | 0.576 | MODERATE |
| Faithfulness (NLI) | −0.173 | −0.185 | WEAK |
| ContextPrecision (bge) | −0.000 | −0.124 | WEAK |
| ContextPrecision (bge reranked) | 0.027 | −0.044 | WEAK |
| ContextPrecision (MiniLM) | −0.039 | −0.018 | WEAK |

Figures: `figures/correlations.png` (bar chart), `figures/comparison_metrics.png`
(scatter + rank-rank per metric).

## 6. Interpretation and caveats

- **Judge saturation**: Faithfulness RAGAS median = 1.000 while NLI median =
  0.000. With a bimodal (and near-degenerate) judge, Spearman is driven by a
  handful of low-judge outliers. The negative sign means the two disagree on
  which answers are grounded — **not** that the shortcut is wrong. Resolution
  requires a 20–30 sample **oracle audit** (human labeling of sentence
  entailment).
- **Ceiling effect**: ContextPrecision shortcuts are ≈ 0.99 mean with tiny
  std → no discriminative power on this corpus. The RAGAS judge is the only
  source of variance, and its variance comes from the retrieved chunks, not
  the ranking. This is a property of the *corpus* (all retrieved chunks are
  relevant), not a general statement about the metric.
- **One judge, one generator**: both RAGAS and the generator use `gemma4:12b`.
  Correlation may be inflated relative to a stronger/cross-model judge.
- **Scale mismatch**: TAS-B dot is unbounded (mean ≈ 98); rank-based
  correlation (Spearman) is scale-invariant, which is why we report it as the
  primary statistic.
- **No examples policy**: per-sample rows and conversation content are
  excluded from this repo by design.

## 7. Compute

- GPU: driver 550.163.01 (CUDA 12.4); system torch built for CUDA 13.0+ →
  CPU fallback for shortcut inference.
- Ollama serves `gemma4:12b` (judge/generator) and `nomic-embed-text`
  (embeddings) on the GPU.
- Shortcut inference models downloaded from HuggingFace (~3.5 GB total):
  bge-reranker-v2-m3, ms-marco-MiniLM-L-6-v2, DeBERTa-v3 MNLI,
  msmarco-distilbert-base-tas-b.
