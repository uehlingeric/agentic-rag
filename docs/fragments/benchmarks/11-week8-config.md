## Week 8 — Final benchmark: full matrix, guardrails on (2026-07-12)

The v0.1.0 canonical run: three providers × four retrieval configs × both pipelines
(24 configs, 46 examples each), executed through the production `GuardedPipeline`
path (runner `--guardrails`) with the week-6 delimiter-hardened prompts
(synthesis.v3, agent-synthesis.v2), dataset v2, judge.v2. Models: Bedrock
`global.anthropic.claude-sonnet-4-6`, Vertex `gemini-3.5-flash`, local
`llama3.1:8b`. The sections above are retained as the historical record of the
build; the tables below supersede them as the headline numbers (the week-4/5 runs
predate the current prompts).
