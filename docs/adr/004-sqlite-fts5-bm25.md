# ADR-004: SQLite FTS5 for BM25 Sparse Retrieval

**Status:** Accepted — 2026-07-04

## Context

The retrieval pipeline (ADR-003) requires a sparse (lexical) retrieval index to
complement dense embeddings. The corpus is ~1,000 chunks across five NIST
documents. Alternatives for BM25 indexing:

- **rank-bm25** (pure-Python library): Stateless, in-memory only, no
  persistence. Every process startup recomputes BM25 statistics from the full
  corpus — O(corpus) initialization cost. No index reuse across runs.
- **Elasticsearch / OpenSearch**: Real external service. Introduces operational
  dependency, deployment overhead, and cloud infrastructure cost unjustifiable
  for a bounded corpus.
- **SQLite FTS5**: Stdlib sqlite3, persistent single-file index, deterministic,
  cross-platform, zero operations overhead.

## Decision

Use SQLite FTS5 (Full Text Search 5, available in sqlite3 since 2015) for BM25
indexing.

**Index layout:** Two tables in a single database file:

1. `chunks(rowid INTEGER PRIMARY KEY, json TEXT)` — full ChunkRecord serialized
   to JSON, with explicit rowid = 1-based insertion order.
2. `chunks_fts(heading, body) USING FTS5` — virtual table with rowid aligned to
   chunks.rowid. Tokenizer: default unicode61 (breaks on whitespace and
   punctuation, case-folds, strips accents).

**Insertion order:** Chunks are inserted in the order provided by the caller
(e.g., file order from the JSONL corpus), making the index deterministic and
byte-stable as far as SQLite allows.

**Query preprocessing:** Raw user queries undergo normalization to neutralize
FTS5 operator injection and handle control IDs (e.g., "AC-2" in NIST
nomenclature):

1. Lowercase the query.
2. Extract token groups matching `[a-z0-9]+(?:-[a-z0-9]+)*` — sequences of
   alphanumeric characters and embedded hyphens (idempotent on the normalized
   query).
3. Each group becomes a double-quoted FTS5 phrase `"group"`. If the group
   contains hyphens, replace them with spaces *inside* the quotes, converting
   it to a phrase query (e.g., "AC-2" → `"ac 2"`, phrase-matching the tokenized
   form of "AC-2" in the index).
4. Join groups with ` OR ` (recall-friendly; FTS5's BM25 variant idf-downweights
   stopwords, so union semantics over control IDs and keywords is safe).
5. If no token groups remain (e.g., query is empty or all punctuation), return
   early with an empty result list.

**Search and scoring:**

- SQL query matches against chunks_fts, ordered by `bm25(chunks_fts) ASC,
  rowid ASC` (FTS5 bm25() returns negative values; smaller is better in the
  ranking).
- Raw scores are inverted to positive: `raw_score = -bm25(...)` (higher is
  better).
- Normalized scores are computed per-query: `score = raw_score / max_raw_in_result_set`
  (top result gets 1.0, others are proportional). Guard against division by zero
  by assigning 0.0 scores if max_raw ≤ 0.
- Each ScoredChunk carries: `rank` (1-based position), `score` (normalized),
  `source_scores: {"bm25": raw_score}` (for hybrid fusion).

## Consequences

**Advantages:**

- Stdlib-only, zero new dependencies; sqlite3 included in Python since 3.2.
- Single-file persistent index, no schema migrations or versioning needed for
  v1.
- Deterministic: same input corpus in same order → identical search results
  across runs and machines.
- Transparent: index is a plain SQLite database; can inspect with `sqlite3 cli`
  or external tools.
- Fast: FTS5 is highly optimized; ~1,000 chunks search sub-millisecond.
- Control-ID handling (AC-2, 3.1.2, etc.) via phrase queries is natural and
  exact.

**Limitations:**

- Scaling beyond ~100k chunks requires database optimization (indexes, query
  planning). For 1,000 chunks, negligible.
- No distributed retrieval; index lives on a single machine.
- FTS5 availability depends on SQLite build (rare in old/embedded builds;
  raises `RuntimeError` if unavailable, with a clear error message).

## Upgrade Path

When the corpus grows or multi-index federation is needed:

1. **10k–100k chunks:** Add explicit FTS5 indexes on `heading` and `body` to
   speed queries. Profile with `EXPLAIN QUERY PLAN` to verify index usage.
2. **100k+ chunks or distributed search:** Migrate to Tantivy (Rust, no
   deployment overhead, one-file persistent index, cross-platform) or managed
   search (e.g., Algolia, Typesense) if team resources allow.
3. **Hybrid pipelines:** If dense+sparse ranking proves insufficient, a learned
   reranker or a simple linear combination of BM25 and embedding similarity can
   be layered on top without changing the index structure.

## Validation

- Control IDs in NIST headings (AC-2, AU-3(1), 3.1.2) are reliably indexed and
  retrieved via phrase queries.
- OR-joined control IDs + keywords (e.g., "account AC-2" → `"account" OR "ac
  2"`) produce intuitive recall without false positives.
- FTS5 injection attempts (SQL operators, unclosed quotes, column syntax) are
  all neutralized by regex-based preprocessing; queries that don't parse as
  FTS5 syntax return empty gracefully (no exceptions).
