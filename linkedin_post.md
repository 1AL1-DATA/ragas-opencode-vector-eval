# metadata
platform: LinkedIn
audience: ML engineers, RAG/LLM evaluation, data engineering
tone: technical but approachable
cta: Repository + methodology in the comments / link below.

---

**Your LLM-judge RAG metrics may be replaceable — and one of them may be lying to you.**

I ran the four classic RAGAS metrics (LLM-as-a-judge, fully local) on a real
agentic-RAG memory system, then compared them against cheap deterministic
shortcuts. Spearman agreement over 99 real query/response pairs:

- **AnswerCorrectness** vs. ROUGE-L + embedding cosine → **ρ = 0.89** (STRONG)
- **AnswerRelevancy** vs. nomic query/document cosine → **ρ = 0.71** (STRONG)
- **Faithfulness** vs. sentence-level NLI → **ρ = −0.18** (judge median 1.000,
  NLI median 0.000 — they flatly disagree)
- **ContextPrecision** vs. graded NDCG → **ρ ≈ 0** (all variants near-constant
  on this corpus)

Three takeaways:
1. The judge cost 8h26m; the shortcuts ran in minutes. Two of four metrics can
   be gated cheaply in CI and only periodically calibrated against a judge.
2. A near-zero correlation is a diagnostic, not a bug — it tells you exactly
   which judge metric to distrust (here: faithfulness needs a human oracle
   audit) and which is corpus-degenerate (here: context precision).
3. Run this shortcut-vs-judge check on any new corpus *before* adopting a RAG
   metric. It's minutes of compute and saves hours of blind trust.

All local: Chroma + nomic-embed-text + gemma4:12b (Ollama) + PyTorch
cross-encoders + DeBERTa-v3 NLI. No conversation content in the artifacts —
only aggregates, figures, and code.

#RAG #LLM #Evaluation #RAGAS #VectorDatabases #LocalAI
