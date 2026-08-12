# Literature digest

Per-paper notes on the works this evaluation builds on. Each entry states the
idea, why it matters here, and how we used (or diverged from) it.

## RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **Venue:** EACL 2023 (findings); arXiv:2309.15217
- **Idea:** decompose RAG quality into reference-free metrics — Faithfulness
  (answer grounded in retrieved context), Answer Relevance (answer addresses
  the question), Context Relevance/Precision/Recall (retrieved context quality)
  — judged by an LLM-as-a-judge with a custom prompt per metric.
- **Why it matters here:** our reference implementation. We use the classic
  `ragas.metrics` classes (Faithfulness, AnswerRelevancy, ContextPrecision,
  AnswerCorrectness) with a fully local judge.
- **How we diverge:** RAGAS assumes a judge (often a frontier model). We
  replace the judge with a local `gemma4:12b` and then ask whether *cheap*
  non-generative scores can reproduce its rankings — an operational question
  RAGAS doesn't answer.

## Chroma
- **What it is:** an open-source embedding database (persistent vector store,
  HNSW index, cosine/dot/IP spaces).
- **Why it matters:** `opencode_sessions` is a Chroma collection (cosine,
  `nomic-embed-text`). The retrieval under evaluation is exactly this store.

## Sentence embeddings / contrastive encoders
- **TAS-B (msmarco-distilbert-base-tas-b):** Hofstätter et al., "Efficiently
  Teaching an Effective Dense Retriever" (SIGIR 2021). Asymmetric dual-encoder
  for query/document retrieval trained with TAS-B (balanced sampling + margin
  distillation).
- **bge-reranker-v2-m3:** BAAI cross-encoder reranker; strong on
  query-vs-document relevance.
- **ms-marco-MiniLM-L-6-v2:** small cross-encoder reranker; our cheap
  alternative.
- **nomic-embed-text:** 768-d matryoshka embedding model, locally served via
  Ollama; with `search_query:`/`search_document:` task prefixes.
- **Why they matter:** the shortcuts are built from these — asymmetric QA
  similarity (TAS-B, nomic) and cross-encoder relevance (bge, MiniLM).

## DeBERTa-v3 MNLI (MoritzLaurer/deberta-v3-base-mnli-fever-anli)
- **What it is:** a Natural Language Inference model fine-tuned on MNLI +
  Fever + ANLI; yields entailment/neutral/contradiction probabilities.
- **Why it matters:** our faithfulness shortcut is sentence-level NLI: is each
  response sentence entailed by any retrieved chunk (probability ≥ 0.5)?

## ROUGE-L
- **Lin (2004), "ROUGE: A Package for Automatic Evaluation of Summaries".**
  Longest-common-subsequence based F1; length-independent, no
  full-summary-overlap requirement.
- **Why it matters:** the lexical half of our AnswerCorrectness shortcut
  (`rouge_l_f1` is a pure-numpy LCS implementation in `src/`).

## NDCG (graded, threshold-free)
- **Järvelin & Kekäläinen (2002), "Cumulated gain-based evaluation of IR
  techniques".** Discounted cumulative gain with graded relevance.
- **Why it matters:** our ContextPrecision shortcut uses *graded* NDCG over
  sigmoid cross-encoder relevance. The binary-threshold alternative collapses
  when ~89% of chunks are lexically relevant (the collapse we measured and
  rejected).

## LLM-as-a-judge
- **Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena"**; Chiang et al. (2024) on position/verbosity bias.
- **Why it matters:** motivates the whole study. Judges are biased and
  saturate; we quantify a saturation case (faithfulness median 1.000) and show
  how cheap agreement analysis surfaces it.

## Verdicts this study leans on
| Metric | Agreement | Reading |
|---|---|---|
| AnswerCorrectness | ρ = 0.89 STRONG | shortcut replaces judge for ranking |
| AnswerRelevancy (nomic) | ρ = 0.71 STRONG | gate on embedding cosine |
| Faithfulness (NLI) | ρ = −0.18 WEAK | judge ceiling; needs oracle audit |
| ContextPrecision | ρ ≈ 0 WEAK | corpus-degenerate here |
