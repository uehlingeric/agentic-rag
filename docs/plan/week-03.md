# Week 3 — Citation-Grounded Generation + Reranking

**Objective:** End-to-end vanilla RAG: query → hybrid retrieval → rerank → synthesized answer with inline citations that resolve to real chunks. By Sunday, `agentic-rag ask "..."` streams a cited answer on any provider, and reranking shows a measured retrieval improvement.

## Exit Criteria

- [ ] `agentic-rag ask "What does NIST require for account management?"` returns an answer with `[1]`-style citations mapping to chunk ids, on all 4 providers
- [ ] Every factual sentence in sampled outputs carries a citation (spot-check 10 queries)
- [ ] Reranking stage on/off is configurable; benchmark table updated with rerank rows
- [ ] Unanswerable golden questions produce an explicit "not found in corpus" response, not a hallucination
- [ ] Streaming works in CLI with citations rendered at the end

## Workstreams

### 1. Reranking
- [ ] LLM listwise reranker: candidates in, relevance-ordered ids out (strict JSON contract, parse-failure fallback to input order)
- [ ] Optional local cross-encoder (`bge-reranker-base`) for the no-API path — behind an extra `[rerank-local]` dependency
- [ ] Config: `rerank: none | llm | cross-encoder`, candidate pool 30 → top 8
- [ ] Extend week 2 eval: nDCG@10 before/after rerank per mode; update `docs/benchmarks.md`

### 2. Answer synthesis
- [ ] Context builder: numbered chunks with doc/section headers, token budget enforcement (drop lowest-ranked first, never truncate mid-chunk)
- [ ] Synthesis prompt: answer only from context; every claim cited `[n]`; explicit contract for "insufficient context" refusals
- [ ] Citation post-processor: validate `[n]` references against provided chunks, strip/flag invalid ones, return structured `Answer{text, citations[], usage}`
- [ ] Streaming: token stream + citation resolution on completion

### 3. Pipeline assembly
- [ ] `RAGPipeline` class composing retriever → reranker → synthesizer; single config object; every stage emits timing + token usage (groundwork for week 7 observability)
- [ ] `ask` CLI command: provider/mode/rerank flags, `--json` output for scripting
- [ ] Integration tests with a stub provider: citation validity, refusal path, token budget edge cases

### 4. Prompt management
- [ ] Prompts as versioned files in `src/agentic_rag/prompts/` (not inline strings) with a loader — evals will reference prompt versions
- [ ] `docs/adr/005-prompt-versioning.md`

## Verification

- 10-query manual review sheet: answer correctness, citation validity, refusal behavior — recorded in `docs/reviews/week-03-spotcheck.md`.
- Run all 5 unanswerable golden questions → 5/5 explicit refusals.
- CI green; pipeline integration tests use stub provider (no keys in CI).

## Commit Milestones (4-6 commits)

1. LLM reranker + eval rows
2. Cross-encoder option
3. Context builder + synthesis prompt + citation validation
4. RAGPipeline + `ask` command + streaming
5. Prompt versioning + spot-check doc

## Risks & Notes

- Citation compliance varies by provider — tune the shared prompt, don't fork per provider; note deltas in the spot-check doc (differences are publishable findings).
- LLM rerank adds latency/cost; capture per-stage timings now so week 4 tables can show quality-vs-cost honestly.
- Do not start agents early — vanilla RAG must be solid; it is the week 5 baseline.
