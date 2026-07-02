# Week 2 — Hybrid Retrieval Core + Golden Dataset v1

**Dates:** Mon Jul 13 – Sun Jul 19, 2026
**Objective:** Working hybrid retrieval over the chunked corpus — BM25, dense vectors, and reciprocal rank fusion — plus the first golden dataset and a retrieval-only eval script. By Sunday there is a **numbers table**: recall@k / MRR / nDCG for BM25 vs dense vs hybrid, committed to `docs/benchmarks.md`.

## Exit Criteria

- [ ] `agentic-rag index` builds both indexes from `chunks.jsonl` deterministically
- [ ] `agentic-rag search "query" --mode bm25|dense|hybrid` returns ranked chunks with scores
- [ ] Golden dataset v1: 30 question/answer/citation triples, human-authored, in `evals/golden/v1.jsonl`
- [ ] `agentic-rag eval retrieval` outputs a metrics table; first benchmark committed
- [ ] Hybrid beats both single modes on recall@10 (if it doesn't, documented analysis of why)

## Workstreams

### 1. Sparse retrieval (BM25)
- [ ] SQLite FTS5 index over chunk text (no extra service dependency — ADR-004 documents choice vs rank-bm25/Elasticsearch)
- [ ] Query preprocessing: lowercasing, punctuation, FTS5 query escaping
- [ ] Top-k API returning normalized scores + chunk metadata

### 2. Dense retrieval
- [ ] FAISS index (IndexFlatIP + normalized vectors to start; note upgrade path)
- [ ] Embedding via `EmbeddingProvider` — default local (Ollama `nomic-embed-text`) so quickstart needs no API key; optional OpenAI/Google embeddings
- [ ] Batch embedding pipeline with progress + resume (embedding 5 NIST docs must be re-runnable)
- [ ] Index persistence: `data/index/faiss.bin` + id-map parquet; `manifest.json` records embedding model + dimensions

### 3. Hybrid fusion
- [ ] Reciprocal rank fusion (k=60 default, configurable)
- [ ] Unified `Retriever` interface: `retrieve(query, mode, top_k) -> list[ScoredChunk]`
- [ ] Tests: fusion math against hand-computed fixtures; determinism test (same query → same ranking)

### 4. Golden dataset v1
- [ ] Author 30 questions across the corpus: 10 lookup ("What does AC-2 require?"), 10 synthesis (cross-section), 5 multi-hop (cross-document), 5 unanswerable (for refusal testing later)
- [ ] Schema: `{id, question, reference_answer, source_citations: [{doc, section}], difficulty, type}`
- [ ] `docs/golden-dataset.md`: authoring criteria, coverage matrix by document and question type

### 5. Retrieval eval script
- [ ] Metrics: recall@k (k=5,10,20), precision@5, MRR, nDCG@10 — judged at section level against `source_citations`
- [ ] Runner: `evals/run_retrieval.py` → markdown table + JSON results in `evals/results/` (timestamped, committed)
- [ ] First benchmark table in `docs/benchmarks.md` with corpus/dataset/config versions noted

## Verification

- Metrics computed against 5 hand-verified fixture queries where correct ranking is known.
- Re-run `agentic-rag index && agentic-rag eval retrieval` twice → identical numbers (determinism).

## Commit Milestones (4-6 commits)

1. FTS5 BM25 index + search
2. Embedding pipeline + FAISS index
3. RRF fusion + unified Retriever + ADR-004
4. Golden dataset v1 + authoring doc
5. Retrieval eval runner + first benchmark table

## Risks & Notes

- Golden dataset authoring is the real bottleneck — timebox to 4 focused hours; quality over count (30 good > 100 sloppy). Week 4 expands to 50.
- Section-level citation matching needs a normalized section-id scheme — align chunker metadata (week 1) if mismatched; fix in chunker, re-ingest.
- If local embedding quality drags dense mode badly, add one API embedding config to the benchmark for contrast — that contrast is itself a publishable finding.
