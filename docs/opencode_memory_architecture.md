# How the vector store manages memory for opencode

This document describes the memory architecture of the system that this study
evaluated: how an agentic coding CLI (opencode) persists its history, how a
local vector database turns that history into searchable memory, and what the
limits of the design are. It is the architectural companion to the
methodology.

## 1. The persistence layer (source of truth)

opencode stores every session in a single SQLite database at
`~/.local/share/opencode/opencode.db`:

| Table | Role |
|---|---|
| `session` | one row per conversation (id, title, directory, model, agent, timestamps) |
| `message` / `part` | the **materialized** conversation content (post-dedupe) — what session resume actually reads |
| `event` | an **append-only** log of streaming updates (`.updated.1` events) — never re-read at resume |

Two properties matter for memory management:

1. **The event table is write-heavy garbage.** In this deployment it grew to
   ~87% of a 5.3 GB database while contributing nothing to resume (the
   projector materializes state into `message`/`part`). It is safe to prune.
2. **Session identity is timestamp-keyed.** Titles and creation/update times
   are the primary human-facing keys (`storage/session_diff` holds 2-byte
   stubs per session). The projector's `SessionUpdated` handler rewrites the
   row, so edits must be made upstream (or re-asserted with a bumped
   `time_updated`), otherwise they are clobbered.

## 2. The vector layer (derived, searchable memory)

Because the SQLite store is query-unfriendly for *semantic* recall, a derived
vector index is built from it:

```
opencode.db
   │  extract.py (reads materialized message/part tables)
   ▼
corpus/corpus.jsonl        # one JSON object per chunk (~1400 chars, overlap 120)
   │  embed.py (Ollama: nomic-embed-text, 768-d, GPU)
   ▼
chroma/ (opencode_sessions)  # cosine, HNSW
   ▲
   └── plugin/memory.ts ── memory_search(query) ──> scripts/search.py (top-k)
```

- **Extraction** reads only the materialized `message`/`part` tables — the
  same tables resume uses — so the vector index inherits exactly what the
  agent can actually remember.
- **Embedding** uses `nomic-embed-text` served on the GPU through Ollama,
  with task prefixes (`search_query:` / `search_document:`) matching the
  evaluation configuration.
- **Query path**: the opencode plugin exposes a `memory_search` tool that
  shells out to `search.py`; results carry session title, date, role, and
  similarity so the agent can cite *when* and *in which session* a decision
  was made.

## 3. The maintenance layer

`~/opencode-vector/maintain/` holds the lifecycle tooling:

- **`cleanup.py`** — deletes frozen/forked sessions with zero unique messages
  and prunes `event` rows to a tail of 500 per session. Safe against opencode
  1.18.15's read paths (verified against source).
- **`swap-db.sh`** — swaps a compacted `opencode.db.compacted` (built by
  cleaning a copy + `VACUUM`) into the live path. Refuses to run while any
  opencode instance is alive, runs `PRAGMA quick_check`, and keeps the
  pre-swap database as a backup.

This decouples *safe maintenance* from *live operation*: the compacted file is
produced from a copy while opencode runs; the swap happens only when the
process is down.

## 4. How this helps memory (and where it doesn't)

**What the design gives you:**

- **Semantic recall.** Ask "how did we set up the evaluator?" and get the
  right session — impossible with a SQL `LIKE` search across 20k messages.
- **Cheap, correct regeneration.** The vector index is a pure function of the
  (cleaned) DB. Corruption or cruft in the source can be repaired once, then
  `extract` + `embed` rebuilds search memory from the clean state — the
  "declot → rebuild" loop.
- **Pruning without data loss.** The append-only log can be dropped; the
  materialized content is what matters.

**Where the limits are (honest caveats):**

- **Still timestamp-keyed at the source.** The vector store is a *derived
  copy*. The authoritative record is SQLite rows keyed by session id and
  times. If the live DB is left at 5.3 GB with frozen forks, the vector index
  rebuilt from it inherits the cruft — memory quality is bounded by store
  hygiene.
- **Rebuild drift.** Between rebuilds, new sessions exist in the DB but not in
  the index. `memory_search` cannot see them until the next extract/embed.
- **One embedding, one model.** `nomic-embed-text` is the retrieval embedding
  and (in this study) the relevancy shortcut — a strong echo between the
  memory system and its evaluation (see the AnswerRelevancy finding, ρ = 0.71).
- **No content isolation in the index.** The corpus and Chroma hold raw
  conversation text on disk. This is fine for a personal local store but is a
  privacy consideration if the directory is ever synced or backed up.

## 5. Relationship to this study

The RAGAS evaluation measured exactly this pipeline: queries were real user
messages; retrieved contexts came from `opencode_sessions` top-5; references
were the historical replies. The ContextPrecision degeneracy (all chunks
relevant) is a direct consequence of the memory system's high-precision
retrieval — the same property that makes `memory_search` useful also makes
that judge metric uninformative on this corpus.
