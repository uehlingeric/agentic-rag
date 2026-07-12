# Week 7 — Observability, API Service, Docker

**Objective:** Production-shape packaging: OpenTelemetry tracing through every pipeline stage, per-request token/cost/latency metrics, a hardened FastAPI service, and a Docker Compose stack that runs the whole system (Ollama included) with no API keys. By Sunday, `docker compose up` serves cited answers locally and traces are visible in Jaeger.

## Exit Criteria

- [ ] Every pipeline stage (guardrails, planner, retrieval, rerank, synthesis, critic) emits OTel spans with token/cost attributes
- [ ] `docker compose up` → FastAPI + Ollama + Jaeger; `curl localhost:8000/ask` returns a cited answer with zero cloud dependencies
- [ ] `/metrics`-style cost summary endpoint: tokens and estimated spend by provider/model/day
- [ ] API has auth (static bearer token minimum), request validation, rate limiting, and OpenAPI docs
- [ ] Load sanity: 20 concurrent requests on the local path — no corruption, coherent p50/p95 latency report

## Workstreams

### 1. OpenTelemetry tracing
- [ ] Span per pipeline stage; attributes: model, tokens in/out, cost, chunk counts, guardrail verdicts, revision count
- [ ] Trace context propagated through LangGraph nodes (graph execution visible as a span tree)
- [ ] Exporters: OTLP (Jaeger in compose) + console for dev; sampling config
- [ ] `docs/observability.md`: span taxonomy, what to look at when answers degrade — written as a runbook, not a feature list

### 2. Metrics & cost tracking
- [ ] In-process metrics registry: request count, error count, stage latency histograms, token/cost counters keyed by provider+model
- [ ] `GET /stats` endpoint + `agentic-rag stats` CLI reading the same store (SQLite)
- [ ] Eval runs also record into the store — one place answers "what did this project cost"

### 3. FastAPI service
- [ ] Endpoints: `POST /ask` (vanilla|agentic, streaming via SSE), `GET /search`, `GET /stats`, `GET /health` (index + provider readiness)
- [ ] Bearer-token auth middleware; pydantic request validation mirroring CLI flags; per-token rate limiting (slowapi); problem+json error shape
- [ ] Service reuses library modules only — zero logic in route handlers (the week 1 discipline pays off here)
- [ ] API integration tests via httpx test client with stub provider

### 4. Docker packaging
- [ ] Multi-stage Dockerfile (slim runtime, non-root user, pinned base digest)
- [ ] `docker-compose.yml`: api + ollama (with model pull init) + jaeger; volumes for corpus/index; healthchecks
- [ ] `make demo`: compose up + ingest + index + sample query — the one-command reviewer path
- [ ] Image builds in CI; smoke test compose stack in CI (ollama CPU small model or stubbed provider profile)

## Verification

- Fresh machine test (or clean VM): `git clone && make demo` → cited answer. Time it; must be under 10 min on broadband (5-min README claim excludes model pull — state that honestly).
- Trace inspection: one agentic request produces a complete, correctly-nested span tree in Jaeger — screenshot saved for week 8 README.
- 20-request concurrency script committed with its latency report.

## Commit Milestones (4-6 commits)

1. OTel spans + Jaeger in dev
2. Metrics store + /stats
3. FastAPI service + auth + tests
4. Dockerfile + compose + healthchecks
5. CI image build + smoke test + observability runbook

## Risks & Notes

- Ollama-in-CI is the flaky point — if unstable by Wednesday, CI uses the stub-provider compose profile and the Ollama path stays a documented manual check.
- Keep the API surface small; this is a reference system, not a product — resist admin endpoints.
- Screenshot/GIF assets captured this week (Jaeger trace, streaming CLI) feed directly into week 8's README work.
