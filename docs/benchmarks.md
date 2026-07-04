# Retrieval Benchmarks

## Methodology

We evaluate retrieval performance against a golden dataset (v1) of 30 security and compliance questions, with 25 answerable and 5 marked unanswerable. Evaluation operates at section granularity: a chunk "covers" a citation if its `doc_id` matches the citation's doc and the citation's section appears in the chunk's `section_ids` list. We compute per-query metrics (recall@k for k∈{5,10,20}, precision@5, MRR@20, NDCG@10) and macro-average them over answerable examples only. NDCG@10 uses novelty-binary gains: a chunk earns gain only when it covers at least one citation not already covered at an earlier rank, so repeat coverage of the same citation (common when a section spans many chunks) cannot inflate the score; IDCG assumes one perfectly relevant chunk per citation at the top — DCG = Σ(rel_i / log₂(i+1)) for i∈1..10, IDCG = Σ(1 / log₂(i+1)) for i∈1..min(|citations|, 10). Evaluation is deterministic: examples and modes are iterated in dataset order and config order, and the index build itself is byte-reproducible (rebuilding from scratch, including a forced re-embed through Ollama, reproduced identical sha256 hashes for `bm25.db`, `faiss.bin`, `id_map.parquet`, and `manifest.json`).

## Week 2 — Retrieval-only (2026-07-04)

Config: corpus `c0a198be…6fd32c` (1,003 chunks, 5 NIST documents), golden dataset v1 (25 answerable), embeddings `nomic-embed-text` (768d, local Ollama) with `search_document:`/`search_query:` task prefixes, BM25 via SQLite FTS5, hybrid = RRF (k=60) over top-50 candidates per mode. Full per-query results: `evals/results/retrieval-20260704-213430Z.json`.

| Mode | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR | NDCG@10 |
|------|----------|-----------|-----------|-------------|-----|---------|
| bm25 | 0.6267 | 0.8600 | 0.9200 | 0.3440 | 0.6106 | 0.6055 |
| dense | 0.7467 | 0.9000 | 0.9200 | 0.4800 | 0.7840 | 0.7382 |
| hybrid | **0.8800** | **0.9000** | **0.9400** | **0.5200** | 0.7147 | 0.7126 |

### Analysis

Hybrid dominates at the cutoffs that matter for a RAG context window: +13.3pp recall@5 over dense (+25.3pp over BM25), and the best recall@20 and precision@5. At recall@10, hybrid ties dense (0.9000) rather than beating it. The tie decomposes to exactly two queries lost and two gained: hybrid drops v1-q16 and v1-q23 from 1.00 to 0.50 — both require SP 800-171 chunks, and BM25's 50-candidate pool floods the fusion with lexically similar SP 800-53 account-management chunks that push the correct SP 800-171 chunk below rank 10 — while gaining v1-q19 and v1-q21, where the two modes' agreement on the right chunks lifts them into the top 10. Dense also keeps the edge on MRR (0.7840 vs 0.7147): when dense's first hit is right, RRF averaging with a noisier BM25 ranking can only move it down. The week-4 dataset expansion (30→50 questions) should tell us whether the cross-document dilution pattern on SP 800-171 queries is systematic; candidate-pool tuning is deliberately deferred until there is a held-out set to tune against.

## Reproduce

```bash
agentic-rag ingest && agentic-rag index && agentic-rag eval retrieval
```

All Week 2 numbers are $0 runs: BM25 is local SQLite, embeddings are local Ollama.
