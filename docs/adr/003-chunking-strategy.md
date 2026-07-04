# ADR-003: Heading-Aware Chunking with Sibling Consolidation, Not Fixed-Size Windows

**Status:** Accepted — 2026-07-04

## Context

The corpus (ADR-002) is five NIST publications with rigid formal structure: control
catalogs (`AC-2`, `AC-2(1)`), dotted section numbering (`3.1.2`), appendices, and
chapter framing. Retrieval quality and citation fidelity both depend on how this
structure is carved into chunks:

- Fixed-size token windows are trivial to implement but cut across control boundaries,
  so a retrieved chunk cannot be cited as "AC-2" — it is "somewhere in the middle of
  the access-control family."
- Pure per-section chunks match citations exactly but produce a pathological size
  distribution: NIST sections range from one sentence (withdrawn controls) to dozens
  of pages (appendix glossaries).

An earlier font-heuristic extraction attempt (headings inferred from font size/bold
alone) shredded body text into ~2,800 pseudo-sections with garbage ids ("three",
"untitled") and a 65-token median chunk — unusable for retrieval.

## Decision

Chunk in two phases: **pattern-based section extraction**, then **sibling consolidation
into token-budgeted windows**.

**Extraction (`ingest/extract.py`).** A line is a heading only if it satisfies both:

1. A structural pattern anchored at line start — control ids
   (`^[A-Z]{2}-\d+(\(\d+\))?` followed by end-of-line or an ALL-CAPS title), dotted
   numerics (`\d+\.\d+…`), `APPENDIX X` / `CHAPTER n`, and AI-RMF category rows.
2. Bold font. Patterns propose; bold confirms — never the reverse.

Both conditions were validated against all five PDFs: every real heading is bold,
while the three false-positive classes that broke the heuristic approach — running
page headers ("APPENDIX A" at 8 pt on every appendix page), errata table rows
("SA-15, SA-16, …", "RA-3”"), and footnote numbers — are all non-bold. TOC entries
are bold but carry dot leaders, so those are rejected explicitly. Section ids obey a
closed grammar (`AC-2`, `AC-2(1)`, `3.1.2`, `appendix-a`, `chapter-3`, `govern-1-1`,
`front-matter`) enforced by test.

**Chunking (`ingest/chunk.py`).** Consecutive sibling sections merge greedily toward a
512-token target, never across top-level units (control *families*, top-level section
numbers, appendices). Oversized units split at sentence boundaries with a 64-token
overlap and a strict 768-token hard cap; hard splits divide evenly rather than leaving
a remainder tail, and windows under 128 tokens are re-packed into a neighbor when the
merge stays under the cap. Merged chunks carry every constituent id in
`section_ids: list[str]` (the retrieval/eval citation key), and `chunk_id` hashes the
per-document ordinal, making ids unique and the whole pipeline byte-deterministic.

## Measured distribution (corpus, o200k_base)

| Metric | Font-heuristic v0 | Final |
|---|---|---|
| Chunks | 2,230+ | 1,003 |
| Median tokens | 65 | 472 |
| Tokens in 256–768 band | ~13% | 92.8% |
| Chunks under 128 tokens | 87% (<256) | 3% |
| Empty chunks | 42 | 0 |
| Chunks over 768 cap | — | 0 |
| Duplicate chunk_ids | 1,337 | 0 |

Per document: sp800-53r5 → 345 sections / 802 chunks, sp800-171r3 → 109 / 131,
ai-rmf → 30 / 57, fips-200 → 3 / 9, fips-199 → 2 / 4.

## Consequences

- Citations resolve to real document structure; the Week-2 golden dataset matches on
  `section_ids`, which survives consolidation by construction.
- Table-heavy chunks are flagged `content_type: "table"` by a column-alignment
  heuristic. Table-*aware* parsing (e.g., AI-RMF GOVERN/MAP subcategory rows as
  distinct sections) is deferred; numbered-section granularity is sufficient for v1.
- The bold-confirmation rule is corpus-validated, not universal. A new document whose
  headings are not bold would need the probe repeated and possibly a per-doc override —
  acceptable for a pinned five-document corpus (ADR-002).
- fips-199/fips-200 headings are un-dotted single numbers ("1 PURPOSE"), which the
  dotted-numeric pattern deliberately ignores (footnote ambiguity); those two small
  documents flow into 2–3 coarse sections. Their chunks stay well-formed via the
  token-budget machinery, so the imprecision costs citation granularity only where the
  documents are least likely to be cited.
