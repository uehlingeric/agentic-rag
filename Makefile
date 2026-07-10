.PHONY: install lint format test test-live ingest chat eval eval-retrieval eval-generation eval-agentic verify-guardrails canary report stats demo

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

# Week-6 guardrail verification: false-positive rate, overhead p50/p95, red-team
# catch rate. No LLM calls, no cost.
verify-guardrails:
	uv run python evals/run_guardrails.py

# Live corpus-poisoning canary (requires a running provider; local path is free).
# Re-records the cassettes that test_canary_playback.py replays in CI.
canary:
	uv run python evals/run_canary.py --provider ollama

report:
	uv run python evals/build_report.py

stats:
	uv run agentic-rag stats

# One-command reviewer path: full local stack (API + Ollama + Jaeger), wait for
# readiness, then one cited answer. First boot pulls ~5 GB of Ollama models and
# ingests + indexes the NIST corpus — allow up to 30 minutes on broadband;
# subsequent runs are seconds.
demo:
	docker compose up -d --build
	@echo "waiting for the API (first boot: model pull + ingest + index)..."
	@timeout 1800 sh -c 'until curl -sf localhost:8000/health | grep -q "\"index_loaded\":true"; do sleep 5; done'
	curl -s -X POST localhost:8000/ask \
		-H "Authorization: Bearer $${AGENTIC_RAG_API__TOKEN:-local-dev-token}" \
		-H "Content-Type: application/json" \
		-d '{"question": "What does control AC-2 require?"}' | python3 -m json.tool

eval: eval-retrieval eval-generation
	uv run python evals/build_report.py \
		--generation-summary evals/results/generation-$(RUN_ID)/summary.json
