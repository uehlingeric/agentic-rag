## Week 5 — Vanilla vs. agentic pipeline (2026-07-10)

Config: retrieval is held fixed at the week-4 winner for cloud providers — hybrid
retrieval + the `llm` listwise reranker (top-8 context) — so the only variable is the
pipeline. `vanilla` is the week-4 single-pass RAG (`synthesis.v2`); `agentic` is the
LangGraph loop (ADR-007): planner (`planner.v1`) classifies direct vs. multi-hop and
decomposes into ≤4 sub-queries, each sub-query retrieves and reranks independently under a
proportional token budget, the synthesizer (`agent-synthesis.v1`) composes across sources,
and the critic (`critic.v1`) triggers ≤2 bounded revisions (`agent-revise.v1`) before
finalizing. Providers: local llama3.1:8b (Ollama), claude-sonnet-4-6 (Bedrock `global.`
profile), gemini-3.5-flash (Vertex, global endpoint). Dataset v2 minus the 4 held-out
planner few-shot items (2 lookup, 2 multi-hop) = 46 questions: 14 lookup, 14 synthesis,
10 multi-hop, 8 unanswerable. Judging: `judge.v2` cross-provider per ADR-006 (no
self-judging); the
week-4 judge-leniency caveat applies unchanged. Full per-row results:
`evals/results/generation-20260710-131027Z/`.
