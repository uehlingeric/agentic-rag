# ADR-005: Prompts as Versioned Files with a Template Loader

**Status:** Accepted — 2026-07-04

## Context

Week 3 introduces the first LLM-dependent behaviors whose quality we measure: listwise
reranking and citation-grounded synthesis. Both are contracts, not prose — the rerank
prompt demands a strict JSON shape the parser depends on, and the synthesis prompt
defines the citation format (`[n]`) and the refusal sentinel (`[NO_ANSWER]`) that the
streaming CLI and citation validator key off. Two failure modes follow if prompts live
as inline string constants:

- **Silent drift.** An "innocent" wording tweak changes measured behavior, and nothing
  ties a benchmark row to the prompt text that produced it. Week-2 retrieval results
  are reproducible because every knob (RRF k, pool, embedding model, corpus
  fingerprint) is recorded; prompt text is a bigger knob than any of them.
- **Escaping pain.** The rerank prompt contains literal JSON braces
  (`{"ranking": [...]}`). With `str.format` every brace needs doubling; with f-strings
  the prompt cannot be a constant at all.

## Decision

Prompts are markdown files in `src/agentic_rag/prompts/` named `{name}.v{N}.md`
(`synthesis.v1.md`, `rerank.v1.md`), shipped as package data and read via
`importlib.resources`, so they load identically from a wheel, an editable install, and
CI. A filename regex (`^[a-z0-9-]+\.v\d+\.md$`) is the whole registry — no manifest to
keep in sync.

`load_prompt(name, version=None)` returns a frozen `Prompt{name, version, text}`;
`version=None` resolves to the highest available so callers get the newest prompt by
default while evals can pin. Rendering uses `string.Template` (`${var}`):

- Literal braces in prompt text need no escaping (the JSON output contract stays
  copy-pasteable).
- `substitute()` raises `KeyError` on a missing variable instead of silently emitting
  a placeholder.
- Zero dependencies; no logic-in-template temptation (loops/conditionals belong in
  Python where they can be tested).

`Prompt.id` (`"synthesis.v1"`) is the attribution key: every eval and benchmark record
that involves an LLM call stores the prompt id(s) alongside model, provider, and corpus
fingerprint.

**Immutability convention.** Once a version has been referenced by a committed result,
its file is never edited — behavior changes create `{name}.v{N+1}.md` and a fresh
benchmark row. Git history then doubles as the prompt changelog, and any historical
result can be re-run against the exact text that produced it.

Alternatives rejected: inline constants (both failure modes above); Jinja2 (a
dependency plus brace/escaping rules, for templating power these prompts do not need);
a prompt-registry service (operational overhead unjustified for a single-repo system —
files + git provide storage, versioning, review, and diffing already).

## Consequences

- Prompt changes are reviewable diffs in PRs, subject to the same review gate as code;
  the citation and refusal contracts are visible in one file rather than assembled
  across string fragments.
- A/B comparison across prompt versions is expressible in eval config (pin v1 vs v2)
  without code changes, which week-5+ agent-vs-vanilla comparisons will use.
- The latest-by-default rule means merging a `v2` file changes runtime behavior
  immediately; anything that must not drift (evals, benchmarks) pins explicitly or
  records `Prompt.id` so the drift is at least attributable.
- One shared prompt per task across all four providers — provider-specific citation
  compliance differences are recorded in spot-check docs as findings, not patched by
  forking prompts per provider (week-3 plan explicitly forbids that).
- The version namespace is per prompt name and flat; at this system's scale (a handful
  of prompts, single-digit versions) that is a feature, not a limitation.
