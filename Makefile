.PHONY: install lint format test test-live ingest chat

install:
	uv venv --allow-existing --python 3.12
	uv pip install -e ".[dev]"

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest

test-live:
	uv run pytest -m live

ingest:
	uv run agentic-rag ingest

chat:
	uv run agentic-rag chat "In one sentence, what is NIST SP 800-53?"
