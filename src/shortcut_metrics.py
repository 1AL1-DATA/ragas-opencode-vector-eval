"""
Local, no-LLM-judge shortcuts for the four RAGAS metrics.

Each shortcut mirrors the semantics of its RAGAS counterpart but uses only
embeddings / cross-encoders / NLI / lexical overlap running on one GPU. No
generative judge is involved, so a full pass costs minutes instead of hours.

Shortcuts and their RAGAS counterpart:
  * context_precision  -> graded NDCG over query-vs-chunk reranker ranking
                          (+ a reranked-order variant)
  * faithfulness       -> sentence-level NLI entailment vs. retrieved chunks
  * answer_relevancy   -> asymmetric QA embedding similarity
                          (TAS-B dot product, plus nomic query/document prefix)
  * answer_correctness -> 0.75 * ROUGE-L-F1(response, reference)
                          + 0.25 * cosine(nomic embed(response), embed(reference))

ContextPrecision relevance is graded (no binary threshold): each chunk's
relevance = sigmoid of a reference-vs-chunk cross-encoder score, and the metric
is graded average precision over the retrieved (and reranked) order. A binary
threshold variant collapses to near-1.0 relevance everywhere on this corpus and
carries no discriminative power (see METHODOLOGY.md).

Output: shortcut_scores.json (meta/aggregates + per-sample scores).

NOTE: this repository does not ship dataset content. `per_sample` entries are
keyed by integer `sample_id` (dataset row order) and contain only numeric
scores, so the file is content-free and reproducible on your own dataset.
"""
import json
import os
import re
import time

import numpy as np
import torch

from . import config

RERANKERS = [
    ("bge_reranker_v2_m3", "BAAI/bge-reranker-v2-m3"),
    ("ms_marco_minilm", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
]
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TAS_B_MODEL = "sentence-transformers/msmarco-distilbert-base-tas-b"

ENTAIL_THRESHOLD = 0.5  # entailment probability cutoff for "sentence is grounded"
NLI_CONTENT_WIN = 0.9  # max NLI premise tokens after truncation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _free(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_sentences(text):
    """Cheap regex sentence splitter (no nltk punkt data needed)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def rouge_l_f1(summary_tokens, reference_tokens):
    """ROUGE-L F1 via LCS over token lists (pure numpy, no deps)."""
    m, n = len(summary_tokens), len(reference_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if summary_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[m, n])
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    f1 = 2 * prec * rec / (prec + rec)
    return float(f1)


def ndcg_graded(relevances, order=None):
    """NDCG over graded relevance scores (0..1), threshold-free.

    Reorder by `order` first if given; 1.0 iff relevance is perfectly ranked.
    """
    rel = np.asarray(relevances, dtype=float)
    if order is not None:
        rel = rel[order]

    def dcg(r):
        return float(np.sum(r / np.log2(np.arange(2, len(r) + 2))))

    idcg = dcg(np.sort(rel)[::-1])
    return dcg(rel) / idcg if idcg > 0 else 0.0


def ollama_embed(texts, model=None, url=None):
    """Batch embedding via the Ollama HTTP API (GPU-side)."""
    import urllib.request

    model = model or config.EMBED_MODEL
    url = (url or config.OLLAMA_URL).rstrip("/")
    req = urllib.request.Request(
        f"{url}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)["embeddings"]


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        print(f"  loading reranker {model_name} on {DEVICE}")
        self.model = CrossEncoder(model_name, device=DEVICE)

    def score_pairs(self, pairs):
        return self.model.predict(pairs, batch_size=32, show_progress_bar=False)

    def context_precision(self, question, reference, chunks):
        """Graded relevance = sigmoid(reference-vs-chunk) — mirrors RAGAS's
        'is this context relevant given the reference' judgment; ordering comes
        from the query-vs-chunk reranker scores. Threshold-free graded NDCG."""
        q_scores = np.asarray(self.score_pairs([(question, c) for c in chunks]))
        ref_scores = np.asarray(self.score_pairs([(reference, c) for c in chunks]))
        rel = 1.0 / (1.0 + np.exp(-ref_scores))
        retrieved = ndcg_graded(rel)
        order = np.argsort(-q_scores)
        reranked = ndcg_graded(rel, order)
        return {
            "retrieved_order": retrieved,
            "reranked_order": reranked,
            "mean_relevance": float(rel.mean()),
        }


def main():
    if not os.path.exists(config.DATASET_JSON):
        raise SystemExit(f"dataset missing: {config.DATASET_JSON}")
    with open(config.DATASET_JSON) as f:
        data = json.load(f)
    print(f"loaded {len(data)} samples")

    # --- 1) ContextPrecision: graded relevance + query-ranking, per sample
    print("\n[context_precision] loading rerankers...")
    rerankers = [Reranker(k) for k, _ in RERANKERS]
    cp_rows = {r.name: {} for r in rerankers}
    for d in data:
        for r in rerankers:
            cp_rows[r.name][d["user_input"]] = r.context_precision(
                d["user_input"], d["reference"], d["retrieved_contexts"]
            )

    # --- 2) Faithfulness: sentence-level NLI vs. chunks
    print("\n[faithfulness] loading NLI model...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE).eval()
    nli_max = int(tok.model_max_length * NLI_CONTENT_WIN)

    def sentence_scores(chunks, sentences):
        pairs = [(c[:nli_max], s) for s in sentences for c in chunks]
        with torch.no_grad():
            logits = nli(**tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)).logits
            probs = torch.softmax(logits, dim=-1)[:, 2]  # entailment
        per_sentence = probs.view(len(sentences), len(chunks)).max(dim=1).values
        return per_sentence.cpu().numpy()

    faith_rows = {}
    for d in data:
        sentences = split_sentences(d["response"])
        if not sentences:
            faith_rows[d["user_input"]] = {"entailed": 0, "n_sentences": 0, "score": 0.0}
            continue
        ent = sentence_scores(d["retrieved_contexts"], sentences)
        faith_rows[d["user_input"]] = {
            "entailed": int(np.sum(ent >= ENTAIL_THRESHOLD)),
            "n_sentences": len(sentences),
            "score": float(np.mean(ent >= ENTAIL_THRESHOLD)),
        }
    _free(nli)

    # --- 3) AnswerRelevancy: TAS-B dot product + nomic query/document cosine
    print("\n[answer_relevancy] loading TAS-B...")
    from sentence_transformers import SentenceTransformer

    tas_b = SentenceTransformer(TAS_B_MODEL, device=DEVICE)
    q_emb = tas_b.encode([d["user_input"] for d in data], batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    a_emb = tas_b.encode([d["response"] for d in data], batch_size=32, show_progress_bar=False, normalize_embeddings=False)
    tas_b_scores = np.sum(q_emb * a_emb, axis=1)
    _free(tas_b)

    print("  nomic prefixed embeddings (search_query:/search_document:)...")
    qq = np.array(ollama_embed(["search_query: " + d["user_input"] for d in data]))
    aa = np.array(ollama_embed(["search_document: " + d["response"] for d in data]))
    nomic_sim = (qq * aa).sum(axis=1) / (np.linalg.norm(qq, axis=1) * np.linalg.norm(aa, axis=1))

    # --- 4) AnswerCorrectness: ROUGE-L-F1 + nomic cosine
    print("\n[answer_correctness] ...")
    rr = np.array(ollama_embed(["search_document: " + d["reference"] for d in data]))
    sem = (aa * rr).sum(axis=1) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(rr, axis=1))
    rouge = [rouge_l_f1(re.findall(r"[a-z0-9]+", d["response"].lower()),
                        re.findall(r"[a-z0-9]+", d["reference"].lower())) for d in data]
    combined = [0.75 * rg + 0.25 * sd for rg, sd in zip(rouge, sem)]

    # --- assemble output: content-free, keyed by sample index ---------------
    per_sample = []
    for i, d in enumerate(data):
        ui = d["user_input"]
        per_sample.append({
            "sample_id": i,
            "context_precision": {
                name: {k: v for k, v in cp_rows[name][ui].items() if k != "scores"} for name in cp_rows
            },
            "faithfulness": faith_rows[ui],
            "answer_relevancy": {"tas_b_dot": float(tas_b_scores[i]), "nomic_query_doc_cosine": float(nomic_sim[i])},
            "answer_correctness": {"rouge_l_f1": rouge[i], "nomic_cosine": float(sem[i]), "combined": combined[i]},
        })

    def avg(key):
        vals = [p["context_precision"][key] for p in per_sample]
        return {k: float(np.mean([v[k] for v in vals])) for k in vals[0] if k in ("retrieved_order", "reranked_order")}

    out = {
        "meta": {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "device": DEVICE,
            "n_samples": len(per_sample),
            "entail_threshold": ENTAIL_THRESHOLD,
            "models": {k: v for k, v in RERANKERS} | {"nli": NLI_MODEL, "tas_b": TAS_B_MODEL},
        },
        "aggregates": {
            "context_precision": {m: avg(m) for m in cp_rows} if per_sample else {},
            "faithfulness": float(np.mean([p["faithfulness"]["score"] for p in per_sample])),
            "answer_relevancy": {
                "tas_b_dot_mean": float(np.mean([p["answer_relevancy"]["tas_b_dot"] for p in per_sample])),
                "nomic_query_doc_cosine_mean": float(np.mean([p["answer_relevancy"]["nomic_query_doc_cosine"] for p in per_sample])),
            },
            "answer_correctness": {
                "rouge_l_f1_mean": float(np.mean([p["answer_correctness"]["rouge_l_f1"] for p in per_sample])),
                "nomic_cosine_mean": float(np.mean([p["answer_correctness"]["nomic_cosine"] for p in per_sample])),
                "combined_mean": float(np.mean([p["answer_correctness"]["combined"] for p in per_sample])),
            },
        },
        "per_sample": per_sample,
    }

    path = os.path.join(config.BASE, "shortcut_scores.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(per_sample)} samples -> {path}")
    print(json.dumps(out["aggregates"], indent=2))


if __name__ == "__main__":
    main()
