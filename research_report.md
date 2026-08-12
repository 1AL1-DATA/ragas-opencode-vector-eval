# RAGAS vs. cheap local shortcuts: an evaluation on a real agentic-RAG memory store

**Author:** Alkiviadis (Adi) Lazaridis  
**Date:** 2026-08-12  
**Status:** research report — aggregate findings only  
**Scope:** 4 RAGAS metrics vs. 9 deterministic shortcut variants, 99 joined
samples, fully local compute

---

## Abstract

LLM-as-a-judge is the default way to evaluate RAG systems, but it is slow,
non-deterministic, and opaque. We evaluate whether cheap, deterministic
shortcuts built from embeddings, cross-encoders, NLI, and lexical overlap can
predict what a local LLM judge would say, on the session store of a real
agentic coding tool (`opencode`) mirrored into a Chroma vector database. We
find that **AnswerCorrectness** is strongly predicted by a
ROUGE-L + semantic-cosine blend (Spearman ρ = 0.89) and **AnswerRelevancy**
by nomic query/document cosine (ρ = 0.71), meaning the expensive judge is
replaceable for ranking. Conversely, **Faithfulness** agreement is weakly
negative (ρ = −0.18) because the judge saturates at a median of 1.000 while
the NLI shortcut reads a median of 0.000, and **ContextPrecision** has no
discriminative signal on this corpus (all variants near-constant). We
conclude that shortcut-vs-judge agreement is a fast, actionable diagnostic for
which RAG metrics are trustworthy on a given corpus — and which judge metrics
themselves need an oracle audit.

## 1. Introduction

Retrieval-augmented generation (RAG) systems are typically evaluated with
LLM-as-a-judge frameworks such as RAGAS. These metrics require a capable judge
model, are non-deterministic, and take hours on a realistic dataset with local
models. For teams that iterate on retrieval pipelines (chunking, embedding
models, ranking, context-window policy), a *deterministic* score that predicts
the judge would be far cheaper to run in CI.

This study answers a narrower, more useful question than "which metric is
best": **on a given corpus, which RAGAS metric can be replaced by which cheap
shortcut, and which metrics are themselves unreliable?**

The subject is not a toy dataset. It is the full session history of an
agentic coding assistant — thousands of real conversations across dozens of
projects — mirrored into a local vector database. The queries are genuine user
requests; the responses are fresh generations conditioned on retrieved chunks;
the references are the historical assistant replies. This is exactly the kind
of messy, production-shaped data on which benchmark conclusions usually fail.

## 2. System under evaluation

The vector store under test (`opencode_sessions`) is derived from a SQLite
session database via a two-stage pipeline:

1. **Extract**: session messages are read from the materialized
   `message`/`part` tables and written to a JSONL corpus, chunked to ~1400
   chars with overlap.
2. **Embed**: the corpus is embedded with `nomic-embed-text` (768-d, cosine)
   on a local GPU via Ollama and stored in Chroma.

Retrieval is top-k cosine search; the generator (`gemma4:12b`, temperature 0)
answers given the top-5 chunks. Section 5 of `METHODOLOGY.md` documents the
architecture in detail.

## 3. Method

### 3.1 Dataset

100 query/reference/context triples sampled with seed 42 (≤5 per session) from
the session store; responses freshly generated at temperature 0 from the
top-5 retrieved chunks. 99 samples survive to the joint analysis.

### 3.2 RAGAS pass

Classic `ragas.metrics` (Faithfulness, AnswerRelevancy, ContextPrecision,
AnswerCorrectness) with a local `gemma4:12b` judge and `nomic-embed-text`
embeddings. Single worker (Ollama is serial); the full run took 8h26m.

### 3.3 Shortcut pass

Deterministic, GPU-computable approximations, designed to mirror each metric's
semantics:

| RAGAS metric | Shortcut | Implementation |
|---|---|---|
| ContextPrecision | graded NDCG | relevance = σ(cross-encoder(reference, chunk)); order = query-vs-chunk reranker (bge / MiniLM), retrieved & reranked |
| Faithfulness | sentence NLI | DeBERTa-v3 MNLI entailment ≥ 0.5 per sentence, max over chunks |
| AnswerRelevancy | TAS-B dot / nomic cosine | asymmetric QA embeddings, prefixed `search_query:` / `search_document:` |
| AnswerCorrectness | ROUGE-L + cosine | 0.75·ROUGE-L-F1 + 0.25·nomic cosine(response, reference) |

The threshold-based ContextPrecision shortcut was rejected after it collapsed
(≈89% of chunks lexically relevant → τ ≈ 0); the graded design restores
discriminative structure.

### 3.4 Agreement analysis

Pearson and Spearman correlation over the 99 joined samples per metric;
verdict bands at |ρ| ≥ 0.7 (STRONG), 0.4–0.7 (MODERATE), < 0.4 (WEAK).

## 4. Results

### 4.1 AnswerCorrectness — replaceable

| Variant | Pearson | Spearman |
|---|---|---|
| 0.75·ROUGE-L + 0.25·nomic | 0.822 | **0.894** |
| ROUGE-L only | 0.788 | **0.870** |
| nomic only | 0.775 | **0.875** |

The combined blend tracks the judge almost perfectly in rank. All three
variants are STRONG; the blend adds little over either alone (0.89 vs
0.87/0.87), so **any** of them suffices for ranking.

### 4.2 AnswerRelevancy — strong with the right embedding

| Variant | Pearson | Spearman |
|---|---|---|
| nomic query/doc cosine | 0.777 | **0.709** |
| TAS-B dot | 0.556 | 0.576 |

The nomic cosine (same embedding family as the retrieval index) is STRONG; the
fine-tuned TAS-B (saturating, unbounded scale) is only MODERATE.

### 4.3 Faithfulness — judge saturates, shortcut cannot be trusted yet

RAGAS median **1.000**, NLI median **0.000**. Spearman −0.185 (WEAK, negative).
The judge calls essentially everything faithful; the NLI shortcut calls most
sentences unentailed by the retrieved chunks. With a bimodal judge, the
correlation is driven by a few low outliers. The correct response is an oracle
audit (human sentence-level entailment labels on 20–30 samples) — not
"trust the shortcut".

### 4.4 ContextPrecision — no signal on this corpus

| Variant | mean NDCG | Spearman |
|---|---|---|
| bge (retrieved) | 0.990 | −0.124 |
| bge (reranked) | 0.991 | −0.044 |
| MiniLM (retrieved) | 0.755 | −0.018 |

Every variant is essentially constant. On this corpus all retrieved chunks are
relevant, so the metric cannot discriminate; the only variance comes from the
RAGAS judge's own chunk judgment. The metric is corpus-degenerate here, not
wrong in general.

### 4.5 Aggregate summary

| RAGAS metric | Shortcut | ρ (Spearman) | Verdict |
|---|---|---|---|
| AnswerCorrectness | combined | 0.894 | STRONG |
| AnswerCorrectness | ROUGE | 0.870 | STRONG |
| AnswerCorrectness | nomic | 0.875 | STRONG |
| AnswerRelevancy | nomic cosine | 0.709 | STRONG |
| AnswerRelevancy | TAS-B | 0.576 | MODERATE |
| Faithfulness | NLI | −0.185 | WEAK |
| ContextPrecision | bge/MiniLM | ≈0 | WEAK |

## 5. Discussion

**For ranking, cheap wins.** Two of four RAGAS metrics are reliably predicted
by shortcuts that run in minutes instead of hours and are fully deterministic.
The implication for CI: gate AnswerCorrectness and AnswerRelevancy on the
shortcuts; run the judge only on periodic calibration runs.

**Disagreement is a diagnostic, not an error.** Where ρ is near zero the
shortcut and judge measure different things. Faithfulness exposes a judge
ceiling; ContextPrecision exposes a corpus degeneracy. Both findings are
actionable: audit the judge with an oracle, or switch metrics.

**Threats to validity.** Single judge and generator (both `gemma4:12b`); a
stronger or differently-prefixed judge could shift the correlations. The
TAS-B shortcut's unbounded scale makes only rank statistics meaningful. The
corpus has a ceiling on context relevance that may not generalize.

## 6. Conclusions

1. AnswerCorrectness and AnswerRelevancy can be evaluated cheaply and
   deterministically on this system.
2. Faithfulness is currently *undecidable* — both judge and shortcut are
   plausible; an oracle audit is the next step.
3. ContextPrecision has no discriminative power on this corpus; do not gate on
   it.
4. The shortcut-vs-judge methodology itself is cheap, and worth running before
   adopting any RAG metric on a new corpus.

## 7. Reproducibility

`src/` contains the full pipeline (`reproduce_eval.py`, `shortcut_metrics.py`,
`compare_eval.py`, `build_aggregates.py`); `tests/` covers the pure functions;
`METHODOLOGY.md` documents every parameter. Data and per-sample rows are
excluded by policy (`data/README.md`).

## References

- Es et al. (2023), *RAGAS: Automated Evaluation of Retrieval Augmented
  Generation* (see `lit_digest.md` for this and other foundational works).
