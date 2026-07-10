# Week 7 — Concurrency Sanity: 20 Concurrent Requests, Local Path

**Date:** 2026-07-10
**Setup:** `agentic-rag serve` on the host (uvicorn, single process), Ollama
0.31.1 serving llama3.1:8b on an RTX 5060 Ti, hybrid retrieval, no rerank,
guardrails and metrics on. One warmup request, then
`uv run python evals/run_load.py --n 20 --concurrency 20` rotating five golden-style
questions. Committed result: `evals/results/load-20260710-212006Z/summary.json`.

## Result

| Requests | Concurrency | OK | Failed | Refusals | p50 (s) | p95 (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 20 | 20 | 0 | 0 | 18.631 | 32.649 | 34.209 |

Integrity checks all pass:

- every response parsed as the full record shape (HTTP 200);
- all 20 `request_id`s unique;
- every non-refusal carried resolved citations;
- the metrics ledger gained exactly one row per request (21 rows including
  warmup, verified via `agentic-rag stats --by source`), with per-stage
  latencies intact.

## Reading the numbers

The p50 of 18.6 s is queueing, not per-request slowness: a single request on
the same setup synthesizes in ~2–4 s, but Ollama serves a small number of
generations in parallel (default `OLLAMA_NUM_PARALLEL`), so 20 simultaneous
requests form a queue and the median request waits for roughly half the fleet
ahead of it. Stage breakdown confirms it — retrieve p95 is 0.35 s and both
guardrail scans are sub-millisecond at p95 even under full concurrency;
synthesis absorbs everything else (p95 32.3 s ≈ the wall clock of the run).

The API layer itself is not the bottleneck at this scale: FastAPI handled 20
in-flight requests on one event loop, the shared retriever (FAISS + SQLite
FTS5 read path) served 20 overlapping retrievals without contention, and the
WAL-mode metrics ledger and audit log took 21 concurrent-ish writes without a
lost or duplicated row.

## What this is not

Not a throughput benchmark. One box, one model server, N=20, one run. The
claim it supports is the week-7 exit bar only: concurrent load on the local
path causes no corruption and produces a coherent latency report.
