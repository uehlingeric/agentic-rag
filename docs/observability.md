# Observability Runbook

How to see what one request did (traces), what the system has been doing
(metrics), and where to look first when answers degrade. Written as a runbook:
each section ends with what to check, not just what exists.

## Turning tracing on

Instrumentation is always compiled in; spans are no-ops until a tracer
provider is installed. Enable via settings (`config.yaml` or env):

```bash
# Dev: spans printed to stderr as JSON
AGENTIC_RAG_OBSERVABILITY__ENABLED=true \
AGENTIC_RAG_OBSERVABILITY__EXPORTER=console \
uv run agentic-rag ask "What does AC-2 require?" --provider ollama

# Jaeger: OTLP/HTTP to localhost:4318 (docker compose publishes it)
AGENTIC_RAG_OBSERVABILITY__ENABLED=true \
AGENTIC_RAG_OBSERVABILITY__EXPORTER=otlp \
uv run agentic-rag ask "What does AC-2 require?" --provider ollama
```

The compose stack (`docker compose up`) enables OTLP export for the API
automatically; the Jaeger UI is at <http://localhost:16686>. `sample_ratio`
head-samples (`ParentBased(TraceIdRatioBased)`) — 1.0 in dev, lower it for
sustained load.

## Span taxonomy

One request produces one trace rooted at `rag.request`. Names and key
attributes are a public contract; tests pin the tree shape.

```
rag.request                     GuardedPipeline — the whole request
├── guardrails.input            PII/injection scan of the question
├── rag.retrieve                vanilla only: hybrid retrieval
├── rag.rerank                  vanilla only: rerank cut
├── rag.synthesize              vanilla only: LLM synthesis + citation check
├── agent.plan                  agentic only: planner decomposition
├── agent.gather                agentic only: per-sub-query retrieval
│   ├── rag.retrieve            one per sub-query (attr agent.sub_query)
│   └── rag.rerank              one per sub-query
├── agent.synthesize            agentic only: draft (re-entered on revision)
├── agent.critic                agentic only: verdict (attr agent.skipped on
│                               refusal drafts)
└── guardrails.output           PII scan of the final answer
```

| Attribute | On | Meaning |
|---|---|---|
| `rag.request_id` | `rag.request` | Joins the trace to the audit record and metrics row |
| `rag.provider`, `rag.model` | `rag.request` | What generated the answer |
| `rag.pipeline` | `rag.request` | `vanilla` \| `agentic` |
| `rag.refusal`, `rag.refusal_reason` | `rag.request` | Machine-readable refusal cause (`input_pii`, `input_injection`, `output_pii`, `out_of_corpus`) |
| `rag.tokens.input/output`, `rag.cost_usd` | request + LLM stages | Usage; cost omitted when unknown (local models) |
| `rag.chunks.count/in/out` | retrieval stages | Candidate pool in, cut size out |
| `rag.citations.invalid_count` | `rag.synthesize` | Markers the model invented |
| `guardrails.detections/blocked/redactions` | guardrail spans | Scan outcome |
| `agent.plan.kind`, `agent.plan.sub_queries`, `agent.plan.fallback` | `agent.plan` | Decomposition; `fallback=true` means the planner reply didn't parse |
| `agent.revision` | `agent.synthesize` | 0 on first draft; increments per critic-triggered rewrite |
| `agent.verdict`, `agent.issues` | `agent.critic` | `pass`/`revise` and issue count |

Streaming note: on `ask_stream` the `rag.synthesize` span closes before the
terminal event is yielded, so consumer-side work never nests under it. SSE
deltas are emitted before `guardrails.output` runs — the terminal result event
carries the scanned answer (ADR-008 documents this limitation).

## Metrics: the request ledger

Every guarded request (CLI `source=cli`, API `source=api`) and every eval LLM
interaction (`eval.generation`, `eval.judge`) appends one row to a SQLite
ledger at `{data_dir}/metrics.db`. Aggregation happens at read time:

```bash
uv run agentic-rag stats                    # by provider
uv run agentic-rag stats --by day --stages  # spend per day + stage latencies
uv run agentic-rag stats --by source --json # cli vs api vs eval, scriptable
curl -H "Authorization: Bearer $TOKEN" localhost:8000/stats?by=model
```

Columns per group: requests, errors, refusals, token sums, cost (None when
every row's cost was unknown), and nearest-rank latency p50/p95. "What did
this project cost" is `stats --by source` — eval rows and interactive rows
share the ledger.

## When answers degrade, look here

**More refusals than usual.** `stats --by day` → refusals column. Then pull a
refusing trace: `rag.refusal_reason` on the root span says *which* layer
refused. `out_of_corpus` with low `rag.chunks.count` on `rag.retrieve` points
at retrieval (index stale? mode misconfigured?); `out_of_corpus` with healthy
retrieval points at synthesis (check `rag.citations.invalid_count` and the
model). `input_*` spikes mean the guardrail policy is catching new traffic —
check the audit log before loosening anything.

**Latency regressions.** `stats --stages` → which stage's p95 moved.
`retrieve` growing → index size or embedding host; `rerank` → the LLM
reranker's provider; `synthesize` → provider/model change or answer-length
creep. For the agentic path, one slow request in Jaeger shows whether time
went to extra sub-queries (`agent.gather` children) or critic revisions
(multiple `agent.synthesize` spans).

**Cost spikes.** `stats --by model` isolates the model; `stats --by source`
separates eval runs from interactive traffic. Eval rows are keyed by
`(request_id, source)` so resumed runs upsert instead of double-counting.

**Citation quality drops.** `rag.citations.invalid_count > 0` on
`rag.synthesize` means the model cited chunks it wasn't shown — correlate
with `rag.chunks.out` (too small a context?) and the prompt version pinned in
the eval records.

**Agentic worse than vanilla.** Known failure mode for small models (see
docs/benchmarks.md, week 5): check `agent.plan.fallback` (planner JSON not
parsing) and `agent.revision` (critic loops on a model that can't act on
critique). The benchmark's guidance stands: don't run the loop on 8B-class
models.

## Joining the three stores

`request_id` is the join key everywhere:

- **Trace** (Jaeger): timing tree + attributes, sampled, ephemeral.
- **Audit record** (`{data_dir}/audit/audit-YYYYMMDD.jsonl`): privacy-scrubbed
  per-request record — verdicts, chunk ids, hashes (docs/audit-log.md).
- **Metrics row** (`{data_dir}/metrics.db`): the aggregate-friendly subset.

A user report ("this answer was wrong at 14:32") resolves as: audit log for
the request_id and chunk ids → Jaeger for where the time went and what the
critic said → `stats` for whether it's one request or a trend.
