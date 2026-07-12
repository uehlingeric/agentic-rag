# Contributing

Issues and PRs are welcome. This is a reference system with published benchmarks, so
the bar for merging is: gates green, no unexplained benchmark drift, honest docs.

## Development setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The local path needs
[Ollama](https://ollama.com) with `llama3.1:8b` and `nomic-embed-text` pulled.

```bash
make install   # uv venv + editable install with dev extras
make test      # unit tests — no network, no API keys
make lint      # ruff check + format + mypy (strict)
```

## Before opening a PR

- All four gates pass locally: `uv run ruff check src tests evals`,
  `uv run ruff format --check src tests evals`, `uv run mypy`, `uv run pytest -q`.
  CI runs them on Python 3.11 and 3.12.
- Unit tests must not touch the network or require API keys. Live provider behavior
  is tested with recorded cassettes or behind the deselected `live` marker.
- Committed prompt files are immutable once a benchmark references them
  ([ADR-005](docs/adr/005-prompt-versioning.md)) — add a new version instead of editing.
- Committed eval results under `evals/results/` are frozen artifacts; regenerating
  them requires paid API calls and a maintainer run. Don't hand-edit them.

## Conventions

- Match the existing style; `ruff format` settles formatting arguments.
- Config flows through `Settings` (pydantic-settings, `AGENTIC_RAG_*` env vars) —
  no ad-hoc `os.environ` reads.
- Library code lives in `src/agentic_rag/`; the CLI and API are thin callers.
  Heavy imports stay inside command bodies to keep CLI startup fast.
- Architectural decisions get an ADR in `docs/adr/`.
