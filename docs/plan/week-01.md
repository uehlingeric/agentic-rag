# Week 1 — Foundations: Scaffold, Provider Adapters, Ingestion

**Objective:** A running skeleton: repo scaffold with CI, a provider adapter layer that swaps Claude/GPT/Gemini/Ollama behind one interface, and an ingestion pipeline that downloads and chunks the NIST corpus. By Sunday, `make ingest` produces a chunked corpus on disk and `make chat` gets a (non-RAG) completion from any configured provider.

## Exit Criteria

- [ ] `pip install -e ".[dev]"` works from a clean clone; CI is green on first PR
- [ ] `agentic-rag chat "hello" --provider ollama|anthropic|openai|google` returns a completion
- [ ] `agentic-rag ingest` downloads NIST PDFs and writes chunked JSONL with metadata
- [ ] Architecture decision records committed for the three foundational choices
- [ ] Every module has at least smoke-level tests; provider adapters fully unit-tested with mocks

## Workstreams

### 1. Repo scaffold & tooling
- [ ] `pyproject.toml`: Python 3.12, src layout (`src/agentic_rag/`), Typer CLI entry point
- [ ] Dev tooling: ruff (lint + format), pytest + pytest-asyncio, mypy on `providers/`
- [ ] `Makefile` targets: `install`, `lint`, `test`, `ingest`, `chat`
- [ ] GitHub Actions: lint + test matrix (3.11, 3.12), badge-ready
- [ ] Config system: `config.yaml` + env-var overrides (API keys never in config), pydantic-settings

### 2. Provider adapter layer
- [ ] `LLMProvider` protocol: `complete()`, `stream()`, `count_tokens()`, structured usage return (input/output tokens, cost)
- [ ] `EmbeddingProvider` protocol: `embed_batch()` with dimension metadata
- [ ] Implementations: Anthropic (Claude), OpenAI (GPT), Google (Gemini), Ollama (local)
- [ ] Cost table per model in `providers/pricing.py` — single source for later cost metrics
- [ ] Retry/backoff wrapper with jitter; provider errors normalized to one exception hierarchy
- [ ] Unit tests with mocked HTTP for every adapter; one live smoke test per provider marked `@pytest.mark.live` (excluded from CI)

### 3. Corpus ingestion
- [ ] Downloader with checksums for: NIST SP 800-53r5, SP 800-171r3, AI RMF 1.0, FIPS 199, FIPS 200
- [ ] PDF extraction (pymupdf); preserve section headings and page numbers
- [ ] Heading-aware chunker with token-length fallback (target 512 tokens, 64 overlap) — chunk metadata: doc id, section path, page, char offsets
- [ ] Output: `data/corpus/chunks.jsonl` + `manifest.json` (doc versions, counts, checksums)
- [ ] Chunker unit tests: heading splits, overlap correctness, token limits

### 4. Architecture records
- [ ] `docs/adr/001-provider-adapter-protocol.md` — why a protocol over LangChain's abstractions at this layer
- [ ] `docs/adr/002-corpus-choice.md` — why NIST publications (public domain, on-brand, heading structure)
- [ ] `docs/adr/003-chunking-strategy.md` — heading-aware vs fixed-size, measured on sample docs

## Verification

- Fresh-clone test: clone to a temp dir, follow README dev-setup section verbatim, confirm all Exit Criteria commands work.
- `make test` ≥ 25 tests passing; CI green on the PR that lands the week's work.

## Commit Milestones (organic cadence — 4-6 commits)

1. Scaffold: pyproject, CI, Makefile, config
2. Provider protocol + Anthropic/Ollama adapters with tests
3. OpenAI/Gemini adapters + pricing table
4. Corpus downloader + PDF extraction
5. Chunker + JSONL output + ADRs

## Risks & Notes

- Gemini SDK churn: pin `google-genai`, wrap thinly.
- PDF extraction quality on 800-53 tables — acceptable to flag table chunks `content_type: table` and defer table-aware parsing (note in ADR-003).
- Keep the CLI thin; all logic in library modules so the API service (week 7) reuses it.
