# ADR-007: Agent Loop Design — Fixed Graph, Bounded Revision, Provider-Native Nodes

**Status:** Accepted — 2026-07-09

## Context

Week 5 adds the agentic layer the project is named for: decompose multi-hop questions,
retrieve per sub-question, synthesize across sources, and self-check citations before
answering. The benchmark question it must serve is *where does agentic RAG beat vanilla
RAG, by how much, and at what cost* — which constrains the design more than any
capability wish-list does:

- **Loops multiply cost.** Every extra LLM call is paid per eval row. An agent whose
  call count varies freely between runs cannot be cost-compared against vanilla.
- **Free-form agency is unevaluable.** If the path through the agent differs per run,
  eval variance swamps the vanilla/agentic delta being measured.
- **The loop must not optimize its own scoring instrument.** Week 4's judge scores are
  the published quality metric; a self-check that shares the judge's prompt would let
  the pipeline train on the test (Goodhart).

## Decision

**Fixed graph, not free-form agency.** The loop is a fixed four-node LangGraph:
`planner → retrieve → synthesize → critic`, with exactly one conditional edge
(critic → synthesize while it says *revise* and the revision cap is not hit, else
finalize). Two alternatives were rejected:

- *ReAct-style free-form loops* — the model decides each step, with tool calls until it
  stops. Unbounded cost, transcripts that cannot be diffed across runs, and no stable
  shape for record/playback testing. The flexibility buys nothing here: the corpus has
  one tool (retrieval) and the useful plans are known shapes.
- *Provider tool-calling loops* — each vendor's function-calling dialect drives the
  loop. This welds orchestration to vendor APIs, exactly what ADR-001's provider
  protocol exists to prevent, and makes cross-provider benchmark rows incomparable.

**Provider-native nodes.** LangGraph supplies state channels, reducers, and conditional
routing — nothing else. Every LLM call inside a node goes through the existing
`LLMProvider` protocol; LangChain model wrappers are not used (ADR-001: one adapter
layer, ours). Swapping the benchmark provider changes zero agent code.

**Classification before decomposition.** The planner first classifies `direct` vs
`multi_hop`; direct questions pass through with the original question as the single
sub-query, so retrieval is identical to vanilla and the agentic overhead on a lookup is
one small planner call — not a pointless decomposition. `multi_hop` yields 2–4
self-contained sub-queries, each retrieved and reranked independently, deduplicated,
and packed under a proportional token budget so no sub-question's evidence is crowded
out. Planner parse failures fall back to a direct plan: a broken planner degrades to
vanilla behavior instead of failing the request.

**Bounded revision with a fail-open critic.** The critic checks four things — every
claim cited, citations actually support their claims, the question fully answered,
cross-source contradictions surfaced — and returns *pass* or *revise* with typed,
actionable issues that are fed back to the synthesizer as a revision turn. At most
`agent.max_revisions` (default 2) rewrites are attempted; after that the answer ships
with a recorded caveat rather than looping. If the critic's reply cannot be parsed
after repair turns, it defaults to *pass*: a broken critic must never burn paid
revision cycles on garbage guidance. Refusal drafts skip the critic entirely — refusals
are handled by refusal-correctness counts (ADR-006), not by revision.

**Critic and judge are prompt-separated (Goodhart).** The critic reuses the judge's
grounding stance (claims are checked against the excerpts the draft cites, never
against reference answers or model memory) and its JSON repair-loop machinery, but is
a separate prompt (`critic.v1`, not `judge.v2`) asking a different question: the judge
scores *how good is this answer* on calibrated 1–5 rubrics; the critic decides *is this
draft shippable and if not what exactly to fix*. Sharing one prompt would let the
pipeline optimize the exact scoring instrument the benchmark publishes. Drift between
critic and judge verdicts is therefore expected, measurable (critic pass-rate vs judge
scores per config), and documented rather than patched away.

**Cost bounds are structural.** Worst case per question: 1 planner call + N sub-query
rerank calls + (1 + max_revisions) synthesis calls + (1 + max_revisions) critic calls —
with defaults ≈ 7 LLM calls against vanilla's 2 (rerank + synthesis). The week-4
pre-run estimate/confirm gate extends to the pipeline dimension so an agentic matrix
never starts without an approved dollar estimate.

**Record/playback instead of mocks.** Provider calls record to FIFO cassettes
(`agent/replay.py`) and play back deterministically, so CI runs the full compiled
graph — loop, cap, reducers — with zero live calls. Playback is order-based, not
content-addressed: prompt edits do not invalidate cassettes.

## Consequences

- Every request is traceable node-by-node (`--trace` dumps each node's input/output),
  and eval rows carry plan kind, sub-query count, revisions, and caveat flags — the
  benchmark can slice deltas by question type.
- The graph shape cannot adapt mid-request: no critic-triggered *re-retrieval* (the
  revision loop can only rewrite from already-gathered evidence). If traces show
  revisions failing for lack of evidence rather than lack of citation discipline,
  a bounded re-retrieve edge is the natural extension — deferred until the benchmark
  proves the need.
- Simple lookups pay one planner call they do not need; the comparative benchmark
  measures this overhead instead of hiding it.
- Two prompts (critic, judge) encode near-identical grounding rules and will drift;
  this is the accepted price of keeping the eval instrument out of the pipeline.
