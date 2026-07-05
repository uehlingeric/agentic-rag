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

## Week 3 — Reranking on/off (2026-07-04)

Config: same corpus, golden dataset, and index as week 2. Each mode retrieves a pool of 30
candidates per query; the baseline row is that pool cut to 10 in retrieval order and the
reranked row is the same pool reordered by the reranker and cut to 10 — identical candidates,
only the ordering differs, so metric deltas isolate the reranker. Metrics are over 10-deep
lists (recall@20 is omitted: it equals recall@10 at this depth). Baseline rows reproduce the
week-2 numbers at matching cutoffs, validating the harness. Rerankers: `llm` = listwise
LLMReranker on ollama/llama3.1:8b (prompt `rerank.v1`, temperature 0); `cross-encoder` =
BAAI/bge-reranker-base via the `[rerank-local]` extra. Full per-query results:
`evals/results/rerank-20260705-011952Z.json` (llm), `evals/results/rerank-20260705-012202Z.json`
(cross-encoder).

| Mode | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@10 |
|------|----------|-----------|-------------|-----|---------|
| bm25 | 0.6267 | 0.8600 | 0.3440 | 0.6075 | 0.6055 |
| bm25+llm | 0.6467 | 0.8000 | 0.3120 | 0.5990 | 0.5921 |
| bm25+cross-encoder | 0.4600 | 0.7267 | 0.1680 | 0.4190 | 0.4315 |
| dense | 0.7467 | 0.9000 | 0.4800 | 0.7840 | 0.7382 |
| dense+llm | 0.7400 | 0.7800 | 0.3920 | 0.7033 | 0.6572 |
| dense+cross-encoder | 0.5533 | 0.8067 | 0.2160 | 0.4761 | 0.5035 |
| hybrid | **0.8800** | **0.9000** | **0.5200** | 0.7147 | **0.7126** |
| hybrid+llm | 0.6400 | 0.8600 | 0.3840 | 0.5527 | 0.5922 |
| hybrid+cross-encoder | 0.4200 | 0.7733 | 0.1520 | 0.4400 | 0.4608 |

### Analysis: both local rerankers hurt — a negative result worth keeping

Neither local reranker improves on the hybrid baseline; both degrade every mode on nearly
every metric, and hybrid — the strongest first stage — is hit hardest (NDCG@10 0.7126 →
0.5922 with the 8B LLM, → 0.4608 with the cross-encoder). The damage is broad, not
concentrated: per-query, hybrid+llm is worse on 16 of 25 questions and better on only 4.

Two failure mechanisms, verified by direct probing rather than assumed:

1. **8B listwise judgment is weak.** The LLM reranker returned parseable rankings (a parse
   fallback would reproduce the baseline row, which is not what we see) — its orderings are
   simply worse than RRF's. Listwise reranking is known to demand strong instruction-following
   models; llama3.1:8b is below that bar.
2. **The cross-encoder confidently prefers the wrong chunks.** Probing v1-q01 (first
   requirement of AC-2): it scores PS-5 and CM-5 chunks 0.97/0.91 and the AC-2 body chunks
   0.63–0.79. The integration is correct (pair order, sigmoid scores, descending sort);
   the preference is the model's. Contributing factors: body chunks open with running-header
   boilerplate ("This publication is available free of charge from: https://doi.org/…") that
   consumes the model's 512-token window, while appendix/crosswalk chunks are keyword-dense;
   and a first stage at recall@5 0.88 leaves a reranker almost no headroom — the classic
   regime where reranking adds variance, not precision.

Cost/latency ($0, local): llm reranking averaged 3.97s/query (128.7k input / 18.8k output
tokens over 75 calls); the cross-encoder averaged 0.59s/query on GPU after model load.

Decisions: `rerank.mode` default stays `none` — the week-5 vanilla-RAG baseline will not
carry a stage that measurably hurts. Boilerplate stripping in chunk text is a week-4
candidate fix (it should help the cross-encoder and shrink synthesis prompts). Rerun this
table with an API-grade model as the listwise judge once provider keys are configured.

## Reproduce

```bash
agentic-rag ingest && agentic-rag index && agentic-rag eval retrieval
agentic-rag eval rerank                             # llm reranker (ollama)
uv sync --extra dev --extra rerank-local
agentic-rag eval rerank --reranker cross-encoder    # local bge-reranker-base
```

All Week 2 and Week 3 numbers are $0 runs: BM25 is local SQLite, embeddings and the LLM
reranker are local Ollama, and the cross-encoder runs locally.
