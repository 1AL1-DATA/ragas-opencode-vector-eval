# Data

## Policy: no examples in this repository

This study evaluates the session history of a personal agentic coding tool.
That history is private conversation content. Accordingly:

- **No dataset, no per-sample rows, no user inputs, responses, references, or
  retrieved chunks are committed.**
- The raw artifacts (`dataset.json`, `shortcut_scores.json`,
  `results/results-*.json`, `comparison.json`) live in the private eval
  workspace (`~/ragas_opencode_vector_db_eval` on the author's machine) and
  are excluded via `.gitignore`.
- This repository ships only aggregate statistics (`results/`), figures,
  and code.

## Provenance of the (private) dataset

| Artifact | Source | License/notes |
|---|---|---|
| `dataset.json` (100 triples) | Sampled from the opencode SQLite session store (`~/.local/share/opencode/opencode.db`), vector store `opencode_sessions` (Chroma, cosine, nomic-embed-text), generator `gemma4:12b` | private; author's own session history |
| `results/results-<ts>.json` | RAGAS run (local judge) on `dataset.json` | private |
| `shortcut_scores.json` | `src/shortcut_metrics.py` on `dataset.json` | private |
| `comparison.json` / `comparison_report.md` | `src/compare_eval.py` | private (per-sample rows excluded from this repo) |

## Reproduction

To reproduce from your own data:

1. Set `EVAL_BASE` to a directory containing your own `dataset.json`,
   `results/results-*.json`, and `shortcut_scores.json` (or regenerate them
   with the pipeline — see `src/reproduce_eval.py`).
2. Rebuild the public aggregates:
   ```bash
   python -m src.build_aggregates
   ```

## `download.sh`

The `download.sh` script in this directory is a placeholder: there is nothing
to download, because the evaluation data is private and excluded by design. It
exists to satisfy the research-template data layer and to document that no
public dataset is bundled.
