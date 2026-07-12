# Week 5 — Agent Loop: Planner → Retriever → Synthesizer → Critic

**Objective:** The agentic layer on LangGraph: query decomposition, multi-step retrieval, synthesis, and a critic that verifies citations and triggers bounded revision. By Sunday the benchmark answers the headline question — **where does agentic RAG beat vanilla RAG, by how much, and at what cost** — with eval numbers on both pipelines.

## Exit Criteria

- [ ] `agentic-rag ask "..." --agentic` runs the full graph on any provider
- [ ] Planner decomposes multi-hop questions into sub-queries; simple questions pass through undivided (no pointless decomposition)
- [ ] Critic catches injected citation errors in fixture tests and triggers ≤2 bounded revisions
- [ ] Benchmark: agentic vs vanilla on dataset v2 — quality, latency, cost side by side
- [ ] Graph state is inspectable: `--trace` dumps every node's input/output for a request

## Workstreams

### 1. Graph skeleton (LangGraph)
- [ ] State schema: original query, plan, sub-query results, draft, critique, revision count, accumulated usage
- [ ] Nodes: `planner`, `retriever`, `synthesizer`, `critic`; conditional edges: critic → revise (≤2) | finalize
- [ ] Provider adapters plugged into nodes via existing protocol (no LangChain LLM wrappers — consistent with ADR-001)
- [ ] Deterministic replay for tests: record/playback of node LLM calls

### 2. Planner
- [ ] Classify then decompose: `direct` (pass through) vs `multi-hop` (2-4 sub-queries)
- [ ] Prompt with few-shot examples from golden multi-hop questions (train/test hygiene: examples must not be eval items — mark them held-out in dataset metadata)
- [ ] Unit tests: lookup questions stay direct; known multi-hop fixtures decompose sensibly

### 3. Retrieval + synthesis integration
- [ ] Per-sub-query hybrid retrieval + rerank; dedupe merged chunk set; token budget across sub-query results (proportional allocation)
- [ ] Synthesizer prompt extended for multi-source composition; citations preserved per source chunk

### 4. Critic
- [ ] Checks: every claim cited, citations support their claims (re-uses judge faithfulness rubric machinery from week 4), question fully answered, contradictions across sources surfaced
- [ ] Output: pass | revise with specific, actionable critique fed back to synthesizer
- [ ] Fixture tests: answers with planted bad citations / unsupported claims must fail; clean answers must pass (no infinite-loop nitpicking — track pass rate on clean fixtures)

### 5. Comparative benchmark
- [ ] Eval runner gains `pipeline: vanilla | agentic` dimension; run both on 2 providers minimum
- [ ] Report section: deltas by question type — hypothesis: agentic wins on multi-hop/synthesis, ties on lookup at 2-4× cost. Publish whatever the data shows.
- [ ] `docs/adr/007-agent-design.md`: why this graph shape, what was rejected (ReAct free-form, tool-calling loops), cost bounds rationale

## Verification

- Trace inspection on 5 multi-hop questions: plans are sensible, retrieval per sub-query is on-topic, critic verdicts justified — recorded in `docs/reviews/week-05-traces.md`.
- Revision loop hard cap proven by test (adversarial fixture that always fails critique → exactly 2 revisions then finalize-with-caveat).
- CI runs graph tests fully on record/playback (no live calls).

## Commit Milestones (4-6 commits)

1. Graph skeleton + state + replay harness
2. Planner + classification tests
3. Multi-source retrieval/synthesis integration
4. Critic + fixture suite
5. Comparative benchmark + ADR-007

## Risks & Notes

- Agent loops multiply cost — enforce the pre-run estimate + confirm gate from week 4 on agentic runs especially.
- If agentic shows no quality win anywhere, that is still a launchable finding — frame as "when agents pay for themselves, measured," which is more credible than universal wins.
- Keep critic rubric shared with the judge but **prompt-separated** — the critic improves answers, the judge scores them; sharing one prompt would let the pipeline optimize for the exact scoring instrument (Goodhart), so document the separation and any drift between them.
