# Summary

Evaluation of an LLM-agent memory / retrieval shortcut suite against RAGAS LLM-judge scores on 99 sampled query/response pairs from a personal opencode session store.

| Metric | RAGAS mean | shortcut mean | Spearman | verdict |
|---|---|---|---|---|
| Faithfulness (NLI) | 0.770 | 0.301 | -0.185 | WEAK |
| AnswerRelevancy (TAS-B) | 0.519 | 97.953 | +0.576 | MODERATE |
| AnswerRelevancy (nomic) | 0.519 | 0.710 | +0.709 | STRONG |
| ContextPrecision (bge) | 0.633 | 0.990 | -0.124 | WEAK |
| ContextPrecision (bge reranked) | 0.633 | 0.991 | -0.044 | WEAK |
| ContextPrecision (MiniLM) | 0.633 | 0.753 | -0.018 | WEAK |
| AnswerCorrectness (combined) | 0.330 | 0.291 | +0.894 | STRONG |
| AnswerCorrectness (ROUGE) | 0.330 | 0.146 | +0.870 | STRONG |
| AnswerCorrectness (nomic) | 0.330 | 0.723 | +0.875 | STRONG |

Headline findings:
1. **AnswerCorrectness** correlates strongly with a cheap ROUGE-L + embedding blend (rho=0.89) — the heavy judge is replaceable for ranking purposes.
2. **AnswerRelevancy** is well-predicted by nomic query/document cosine (rho=0.71, STRONG); TAS-B is moderate.
3. **Faithfulness** agreement is weak and negatively signed (rho=-0.18): the RAGAS judge saturates at 1.0 (median 1.000) while the NLI shortcut reads low (median 0.000) — see METHODOLOGY for the oracle-audit caveat.
4. **ContextPrecision** shows no discriminative agreement (rho~0): all variants are near-constant on this corpus, so the metric cannot rank systems here.

See `comparison_report.md` in the source eval workspace and `docs/opencode_memory_architecture.md` for the architectural context.

