# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [1.0.0] — 2026-07-12

### Changed

- Module and CLI interfaces are now covered by stable SemVer

### Removed

- `CONTRIBUTING.md` and issue templates — the project is a reference system showcase, not a contribution-seeking package
- Explicit calendar dates from the weekly plan docs; the plan reads as sequence and scope

## [0.1.0] — 2026-07-12

First public release: the full 8-week build (plans in `docs/plan/`, decisions in
`docs/adr/`).

### Added

- **Ingestion & retrieval** — NIST corpus (SP 800-53r5, SP 800-171r3, AI RMF,
  FIPS 199/200) chunked with section-aware boundaries; hybrid retrieval (SQLite
  FTS5 BM25 + FAISS dense + reciprocal rank fusion); optional LLM and
  cross-encoder rerankers.
- **Providers** — Anthropic (API/Bedrock), Google (API/Vertex), OpenAI, Ollama
  behind one adapter protocol with retry, pricing, and usage accounting; a
  deterministic stub provider for CI.
- **Vanilla pipeline** — citation-grounded synthesis with validated `[n]`
  markers, refusal on out-of-corpus questions, streaming.
- **Agentic pipeline** — LangGraph planner → gather → synthesize → critic loop
  with bounded revisions; per-stage trace output.
- **Evals** — golden dataset v2 (50 items, 4 held out); retrieval metrics
  (recall@k, MRR, nDCG); LLM-as-judge generation scoring (faithfulness,
  relevance, citation accuracy) with a cross-judge calibration study and
  published judge-bias caveats; committed benchmark runs and a generated
  `docs/benchmarks.md`.
- **Guardrails** — always-on PII scanner (regex + optional spaCy NER),
  prompt-injection screening with excerpt delimiters, per-entity refusal
  policy, schema-versioned JSONL audit log; red-team suite with annotated
  known misses.
- **Observability** — OpenTelemetry span per pipeline stage with token/cost
  attributes; per-request SQLite metrics ledger with `agentic-rag stats`.
- **API** — FastAPI service: `POST /ask` (vanilla/agentic, SSE), `GET /search`,
  `GET /stats`, `GET /health`; bearer auth, per-token rate limits, RFC 9457
  problem+json errors; guardrails non-bypassable over HTTP.
- **Packaging** — two-stage Dockerfile (non-root, digest-pinned base);
  docker-compose stack (API + Ollama + Jaeger) with `make demo`; CI smoke
  profile with the stub provider.
- **CI** — ruff + mypy (strict) + pytest on Python 3.11/3.12, Docker smoke,
  pip-audit, gitleaks, CodeQL.

[0.1.0]: https://github.com/uehlingeric/agentic-rag/releases/tag/v0.1.0
