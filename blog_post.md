# Your LLM-judge RAG metrics are slower than the shortcuts that can replace them

*The 8-hour test you didn't need to run — and the two metrics your judge can't actually judge.*

If you build RAG systems, you've met the workflow: run RAGAS with an
LLM-as-a-judge, wait hours, get a number, wonder if it means anything. I ran
that workflow on a real agentic-RAG memory system — the session store of an
LLM coding agent mirrored into a local vector database — and then asked a
different question than "which metric is best":

> Which of these expensive judge metrics can a cheap, deterministic formula
> predict — and which ones is the judge itself getting wrong?

The results surprised me. Here's the short version.

## What I did

- A local model (`gemma4:12b`) judged 100 real query/response pairs across the
  four classic RAGAS metrics. It took **8 hours and 26 minutes**.
- I then computed nine "shortcut" scores for the same samples — the kind of
  thing you can run in minutes on one GPU: sentence-level NLI for
  faithfulness, asymmetric QA embeddings for answer relevancy, graded NDCG
  with a cross-encoder reranker for context precision, and ROUGE-L + embedding
  cosine for answer correctness.
- I measured how well each shortcut predicts what the judge said (Spearman
  rank correlation over 99 joined samples).

## What I found

**1. AnswerCorrectness is replaceable — ρ = 0.89.**

A blend of 75% ROUGE-L and 25% semantic cosine tracks the judge almost
perfectly in rank. Same for ROUGE alone (0.87) and cosine alone (0.87). You
can gate your CI on the cheap score and reserve the judge for occasional
calibration runs.

**2. AnswerRelevancy — strong with the right embedding (ρ = 0.71).**

The same embedding family the retrieval index uses (`nomic-embed-text`) with a
`search_query:`/`search_document:` prefix reproduces the judge's ranking
strongly. A fine-tuned TAS-B encoder only got moderate agreement (0.58) and
saturates numerically.

**3. Faithfulness — the judge is lying (ρ = −0.18).**

RAGAS gave a median score of **1.000** — everything is faithful! The NLI
shortcut gave a median of **0.000** — nothing is entailed by the retrieved
chunks! Two opposing verdicts, and the correlation is weakly *negative*. One
of them is lenient, one strict, and neither can be trusted until you audit
~30 samples by hand.

**4. ContextPrecision — no signal on this corpus (ρ ≈ 0).**

Every variant was near-constant: bge-reranker NDCG ≈ 0.99. All retrieved
chunks are relevant here, so the metric physically cannot rank anything. It's
not that the metric is broken — it's that the corpus has no floor.

## What this means for you

- **Run the shortcut-vs-judge check before you adopt any RAG metric on a new
  corpus.** It costs minutes and tells you which metrics are measuring
  something real.
- **Cheap beats slow for ranking.** Deterministic shortcuts don't drift, don't
  need a judge model loaded, and run in CI.
- **A zero correlation is a gift.** It tells you exactly which judge metric to
  distrust — and where to spend your audit budget.

The whole methodology, scripts, and aggregate tables are in the companion
repository. No conversation content is shipped — only numbers, figures, and
code.

*Built entirely with local models: Chroma + nomic-embed-text + Ollama
(gemma4:12b), PyTorch cross-encoders, and DeBERTa-v3 NLI.*
