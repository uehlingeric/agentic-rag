# ADR-001: Provider Adapters via a Native Protocol, Not LangChain Abstractions

**Status:** Accepted — 2026-07-04

## Context

The system must run identically across Claude, GPT, Gemini, and Ollama: completions,
streaming, token/cost accounting, and embeddings. The obvious shortcut is LangChain's
`BaseChatModel`/`Embeddings` hierarchy, which already wraps every vendor.

## Decision

Define our own minimal contracts in `providers/base.py` — `LLMProvider` and
`EmbeddingProvider` as `typing.Protocol`s plus frozen dataclasses (`Message`,
`Completion`, `Usage`, `StreamEvent`) and a normalized exception hierarchy — and
implement one thin adapter per vendor on the official SDKs (raw `httpx` for Ollama).

Key contract choices:

- **Structural typing (`Protocol`)** over ABC inheritance: adapters stay dependency-free
  toward our framework; tests can substitute stubs without registration ceremony.
- **`Usage` on every completion**, with `cost_usd` resolved from a single pricing table
  (`providers/pricing.py`). Cost tracking is a first-class launch metric; it cannot be
  bolted on later.
- **Normalized exceptions** with an explicit `retryable` flag, so one retry/backoff
  wrapper (`providers/retry.py`) serves every vendor and downstream code never imports
  vendor exception types.
- **System prompt as a parameter**, not a message role — vendors disagree on the wire
  format (Anthropic separates it; OpenAI in-lines it); the contract picks the semantic
  and adapters translate.
- **Streaming as `AsyncIterator[StreamEvent]`** ending with a final event that carries
  the assembled `Completion` + usage — callers get deltas for UX and exact accounting
  without a second code path.

## Why not LangChain here

- **Abstraction depth:** LangChain's model classes carry callback managers, serialization,
  run trees, and config plumbing we don't use. When (not if) a vendor call misbehaves,
  debugging through those layers costs more than the adapters saved.
- **Interface churn:** the LangChain surface has historically moved fast; pinning our
  contract to five dataclasses we own removes an entire class of upgrade risk for a
  repo meant to stay reproducible after launch.
- **Cost/usage fidelity:** uniform, per-call token+cost accounting across four vendors
  is central to the published benchmarks; owning the adapter means owning that math.
- **Honest layering:** LangGraph is still used for agent orchestration (week 5) — this
  decision is about the *model-call* layer only. LangGraph nodes call our providers
  through the same protocol (revisited in ADR-007).

## Consequences

- We maintain four adapters (~100–200 lines each) and their mocked tests.
- New vendors cost one module + tests; nothing downstream changes.
- SDK-specific features (prompt caching, JSON mode variants) must be exposed through
  the protocol deliberately, or not at all — a feature gate, which for a reference
  system is a benefit: the surface stays explainable.
