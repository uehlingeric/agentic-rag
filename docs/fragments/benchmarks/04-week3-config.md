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
