# ADR-002: NIST Publications as the Reference Corpus

**Status:** Accepted — 2026-07-04

## Context

A RAG showcase needs a corpus that is legally redistributable, large enough to make
retrieval non-trivial, structured enough to ground citations, and domain-relevant to
the systems this repo is meant to demonstrate (governed, auditable deployments).

## Decision

Use five NIST publications: SP 800-53 Rev. 5, SP 800-171 Rev. 3, AI RMF 1.0
(NIST.AI.100-1), FIPS 199, and FIPS 200.

## Rationale

- **Public domain.** U.S. federal government works — no licensing risk in a public
  repo that ships the chunked corpus derivatives and cites full text in eval fixtures.
- **On-brand domain.** Security controls, risk management, and AI governance mirror
  the federal/compliance settings where guardrailed, audited RAG actually gets
  deployed; the week 6 guardrail policies reference the same vocabulary.
- **Citation-friendly structure.** Deep, consistent heading hierarchies and stable
  identifiers (control IDs like `AC-2`, numbered sections, appendices) give a natural
  section-level citation scheme — the golden dataset can specify ground truth as
  `{doc, section}` pairs and retrieval can be scored against it mechanically.
- **Real retrieval difficulty.** ~1,000+ pages across five documents with heavy
  cross-referencing (800-53 ↔ 800-171 overlap, AI RMF cross-walks) — enough for
  meaningful multi-hop questions, distinguishable BM25-vs-dense behavior, and honest
  unanswerable questions.
- **Stable, checksummable sources.** Fixed-version PDFs on nvlpubs.nist.gov; the
  ingestion manifest pins URLs + SHA-256, so published benchmark numbers stay
  reproducible.

## Alternatives rejected

- **Wikipedia dumps:** license fine, but generic; no stable section-id scheme;
  benchmark results would say little about governed-document RAG.
- **Company docs / scraped sites:** licensing and stability problems; unreproducible.
- **Academic paper sets (arXiv):** redistribution is per-paper; PDFs are layout-hostile;
  domain diverges from the guardrails story.

## Consequences

- PDF extraction must handle NIST's layout (running heads, dense tables in 800-53).
  Table-aware parsing is deferred; table-ish chunks are flagged (`content_type:
  "table"`) — see ADR-003.
- The corpus is version-pinned; a NIST revision (e.g. 800-53 Rev. 6) is a deliberate,
  manifest-versioned upgrade, never an implicit drift.
