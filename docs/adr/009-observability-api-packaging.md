# ADR-009: Observability, API Service, and Packaging — Production Shape Without Product Scope

**Status:** Accepted — 2026-07-10

## Context

Week 7 gives the system its production shape: OpenTelemetry tracing through every
pipeline stage, a per-request metrics ledger, a hardened HTTP service, and a Docker
Compose stack that serves cited answers with zero cloud dependencies. The constraints
carried in from earlier weeks:

- **The library is the product; interfaces are thin.** The week-1 rule (CLI commands hold
  no logic) must survive contact with an HTTP framework — the API can add auth, limits,
  and serialization, never behavior.
- **Benchmarks must stay comparable.** Instrumentation cannot perturb what weeks 4–6
  measured, and the eval runner keeps bypassing guardrails while everything else cannot.
- **The quickstart stays keyless.** `docker compose up` has to reach a cited answer with
  no API keys, and CI has to smoke the stack with no models at all.
- **This is a reference system, not a product.** No admin endpoints, no user accounts, no
  metrics dashboard — the surface is exactly what a reviewer needs to see production
  thinking.

## Decision

**Tracing: always compiled in, no-op until enabled.** Every stage wraps its work in a
span from a module tracer; `setup_tracing()` installs a real provider (console or
OTLP/HTTP for Jaeger) only when `observability.enabled` is set. No decorators or
middleware frameworks — inline `with` blocks at the exact boundaries the timings already
measure. The span taxonomy (`rag.request` root; `guardrails.*`, `rag.*`, `agent.*`
children; token/cost/refusal attributes) is a documented contract pinned by span-tree
tests. Two subtleties are load-bearing: the streaming synthesize span closes *before* the
terminal event is yielded so consumer-side spans nest under the request root, and the
console exporter writes to stderr so `--json` output stays parseable.

**Metrics: an append-only SQLite ledger, aggregation at read time.** One row per guarded
request and per eval LLM interaction, keyed `(request_id, source)` so resumed eval runs
upsert instead of double-counting. No in-process histogram state — WAL mode plus a fresh
connection per operation makes concurrent API workers safe, and `agentic-rag stats` (and
`GET /stats`) computes counts, token/cost sums, and nearest-rank percentiles with plain
SQL + Python at query time. The alternative — a metrics library with registries and
exporters — adds a dependency and a second source of truth for numbers the audit log
already carries; a ledger the reviewer can open with `sqlite3` is more legible.

**API: FastAPI over the same library paths, refusals are not errors.** `POST /ask`
constructs the same objects the CLI does and returns the same record shape the CLI
`--json` flag emits (pinned by tests on both sides). Guardrails cannot be bypassed over
HTTP — only the eval runner constructs bare pipelines. A guardrail block or out-of-corpus
refusal is a **200** with `refusal_reason` set: the system worked as designed, and
callers need the machine key, not an exception. problem+json (RFC 9457) is reserved for
transport failures: 401 auth, 422 validation, 429 rate limit, 502 provider outage, 500
unexpected. Auth is a static bearer token (`hmac.compare_digest`, bound at app
construction); rate limiting is slowapi keyed per token per route. Reranker and pipeline
objects are built per request because `Reranker.last_usage` is call-scoped mutable state
(the week-5 gotcha) — only the retriever (FAISS + BM25 + embedder) is shared. SSE
streaming inherits the CLI's documented limitation: deltas are emitted before the output
guardrail scan; the terminal `result` event carries the scanned answer (ADR-008).

**Packaging: one image, first-boot bootstrap, a stub-provider smoke path.** A two-stage
Dockerfile (uv builder → slim runtime, digest-pinned bases, non-root user) produces an
image whose entrypoint auto-runs ingest + index when the data volume is empty, so
`docker compose up` alone ends at cited answers — model pull and first-boot indexing are
the honest cost, stated in the README rather than hidden. The compose stack is
api + Ollama (with a one-shot model-pull init container) + Jaeger. CI cannot pull 5 GB of
models, so a deterministic **stub provider** (canned cited answer, seeded 64-dim
embeddings) registered behind the normal provider registry powers a `smoke` compose
profile over a small committed fixture corpus — the smoke test exercises image build,
index build, auth, guardrails, and `/ask` end to end with zero models and zero egress.

## Consequences

- One request is observable three ways joined by `request_id`: trace (timing tree),
  audit record (privacy-scrubbed verdicts), metrics row (aggregates). The runbook
  (docs/observability.md) documents the join and the degradation playbook.
- Eval cost now lands in the same ledger as interactive traffic; "what did this project
  cost" is one query. Benchmark numbers are unaffected: the eval runner still bypasses
  guardrails, and instrumentation adds no measurable overhead to the stages it wraps.
- The API surface is deliberately small (ask/search/stats/health). Anything more —
  index management, corpus upload, key rotation — is product scope this reference
  system refuses.
- The stub provider is a permanent test seam: any future pipeline change is smokeable
  in CI without models, and the fixture corpus keeps the smoke honest (real chunks,
  real citations).
- `from __future__ import annotations` is banned in `api/app.py` specifically: FastAPI
  resolves stringified route annotations against module globals, and the auth dependency
  is a closure-local. The module docstring says so to stop a future cleanup from
  reintroducing the bug.
