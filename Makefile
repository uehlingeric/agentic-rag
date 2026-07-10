.PHONY: install lint format test test-live ingest chat eval eval-retrieval eval-generation eval-agentic report

# One RUN_ID per make invocation so both eval-generation passes share a results dir.
RUN_ID ?= $(shell date -u +%Y%m%d-%H%M%SZ)

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

eval-retrieval:
	uv run python evals/run_retrieval.py

# Full week-4 matrix: {bm25, dense, hybrid} × rerank none, plus hybrid+llm-rerank.
# Prints a cost estimate and asks for confirmation before any paid call.
eval-generation:
	uv run python evals/run_generation.py --run-id $(RUN_ID) \
		--provider ollama --provider anthropic --provider google \
		--mode bm25 --mode dense --mode hybrid --rerank none
	uv run python evals/run_generation.py --run-id $(RUN_ID) \
		--provider ollama --provider anthropic --provider google \
		--mode hybrid --rerank llm

# Week-5 comparative benchmark: vanilla vs agentic on the best retrieval config.
# Prints a cost estimate and asks for confirmation before any paid call.
eval-agentic:
	uv run python evals/run_generation.py --run-id $(RUN_ID) \
		--provider ollama --provider anthropic --provider google \
		--mode hybrid --rerank llm \
		--pipeline vanilla --pipeline agentic

report:
	uv run python evals/build_report.py

eval: eval-retrieval eval-generation
	uv run python evals/build_report.py \
		--generation-summary evals/results/generation-$(RUN_ID)/summary.json
