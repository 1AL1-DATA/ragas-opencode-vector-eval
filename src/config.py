"""Shared configuration for the RAGAS-vs-shortcut evaluation tooling.

This repository ships the *analysis* code and aggregate results only. The raw
per-sample artifacts (`dataset.json`, `shortcut_scores.json`, `results/*.json`)
contain private conversation content and are intentionally NOT committed. Point
the paths below at your own copies when you want to reproduce the aggregates.
"""
import os

# --- vector store (subject under evaluation) --------------------------------
CORPUS = os.path.expanduser(os.environ.get("CORPUS", "~/opencode-vector/corpus/corpus.jsonl"))
CHROMA_DIR = os.path.expanduser(os.environ.get("CHROMA_DIR", "~/opencode-vector/chroma"))
COLLECTION = os.environ.get("COLLECTION", "opencode_sessions")

# --- models (all local) -----------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gemma4:12b")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma4:12b")

# --- sampling (when rebuilding the evaluation dataset) ----------------------
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
TOP_K = int(os.environ.get("TOP_K", "5"))
SEED = int(os.environ.get("SEED", "42"))
MAX_PER_SESSION = int(os.environ.get("MAX_PER_SESSION", "5"))
Q_MIN_LEN = int(os.environ.get("Q_MIN_LEN", "15"))
Q_MAX_LEN = int(os.environ.get("Q_MAX_LEN", "1500"))
ANSWER_MAX_TOKENS = int(os.environ.get("ANSWER_MAX_TOKENS", "1200"))
DISABLE_THINK = os.environ.get("DISABLE_THINK", "1") == "1"
OLLAMA_REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "240"))
CONTEXT_CHUNK_CHARS = int(os.environ.get("CONTEXT_CHUNK_CHARS", "1000"))
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "6h")

# --- paths ------------------------------------------------------------------
# The evaluation happens against private data OUTSIDE this repo.
BASE = os.environ.get("EVAL_BASE", os.path.expanduser("~/ragas_opencode_vector_db_eval"))
DATASET_JSON = os.path.join(BASE, "dataset.json")
RESULTS_DIR = os.path.join(BASE, "results")
SHORTCUT_JSON = os.path.join(BASE, "shortcut_scores.json")
COMPARISON_JSON = os.path.join(BASE, "comparison.json")

# This repo's aggregate outputs.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGGREGATES_CSV = os.path.join(REPO, "results", "aggregates.csv")
CORRELATIONS_CSV = os.path.join(REPO, "results", "correlations.csv")
SUMMARY_MD = os.path.join(REPO, "results", "SUMMARY.md")
FIGURES_DIR = os.path.join(REPO, "figures")
