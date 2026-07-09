## Week 2 — Retrieval-only (2026-07-04)

Config: corpus `c0a198be…6fd32c` (1,003 chunks, 5 NIST documents), golden dataset v1 (25 answerable), embeddings `nomic-embed-text` (768d, local Ollama) with `search_document:`/`search_query:` task prefixes, BM25 via SQLite FTS5, hybrid = RRF (k=60) over top-50 candidates per mode. Full per-query results: `evals/results/retrieval-20260704-213430Z.json`.
