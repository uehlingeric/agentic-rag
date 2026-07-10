# Week 5 Trace Inspection: Agent Loop on 5 Multi-Hop Questions

**Date:** 2026-07-09
**Config:** `agentic-rag ask --agentic --trace` — planner.v1 → hybrid retrieval per sub-query
(pool 30) → llm rerank (top 8) → proportional-budget merge (≤6000 tokens) →
agent-synthesis.v1 → critic.v1, `max_revisions=2`. Provider: ollama / llama3.1:8b,
temperature 0 (the zero-cost path; cloud traces will accompany the comparative benchmark).
Questions: the five hardest non-held-out multihop items (v1-q23, v1-q24, v1-q25, v2-q39,
v2-q43). Every plan, per-sub-query chunk list, draft, and critic verdict below was reviewed
by hand against the trace dumps and the corpus chunk text.

## Headline

The **loop plumbing is correct end to end** — 5/5 questions classified `multi_hop` with 2
sub-queries each, zero planner parse fallbacks, per-sub-query retrieval + rerank + dedupe +
budget merge produced 11–12 chunk contexts at 5.1–5.9k tokens (under the 6k cap), no invalid
citation markers, no revision-cap caveats. The **quality ceiling on an 8B model is visible
in three places**: one planner few-shot anchoring failure, a lenient critic that passed two
flawed drafts, and one new sentinel-misuse variant. All three are exactly what the
comparative benchmark and trace instrumentation exist to surface.

## Plans: 9 of 10 sub-queries well-formed; one anchoring failure

Four questions decomposed exactly as a human would — e.g. v2-q43 into *"SP 800-171 Revision 3
requirement 3.1.8 handling of unsuccessful logon attempts"* + *"SP 800-53 Revision 5 control
AC-11 device access requirements"*: self-contained, publication-explicit, one per fact
source.

The failure is instructive. **v1-q24** asks how AI RMF governance complements *"security
controls for protecting AI system resources in SP 800-53"*. Sub-query 1 was perfect (*"NIST
AI RMF AI system governance measures"* — retrieval returned exclusively ai-rmf chunks).
Sub-query 2 came back *"SP 800-53 Revision 5 control AC-2 account management requirements"*
— **copied from the planner few-shot examples rather than adapted to the question**, which
never mentions AC-2. Three of the four few-shot decompositions in planner.v1 name AC-2; an
8B model pattern-matched instead of generalizing. Downstream, the answer discussed
IA-4/CM-5/CA-7 (whatever the AC-2 rerank surfaced) and never delivered the asked-for
comparison. Candidate planner.v2 change: diversify the few-shot target controls. Worth
re-measuring on cloud models before touching the prompt — this may be an 8B-only failure.

## Per-sub-query retrieval: on-topic where the corpus cooperates

- **On-topic:** v2-q43 sub-query 1 put `sp800-171r3:03.01.08` at rank 1; v1-q24 sub-query 1
  returned only ai-rmf chunks; every AC-2-shaped sub-query put `sp800-53r5:AC-2` at rank 1.
- **Corpus-shaped noise:** FIPS-directed sub-queries (v1-q25, v2-q39) surfaced SP 800-53's
  RA-2/PL-10/PL-11 (categorization and baseline-selection *controls*) above the FIPS
   199/200 source text, which in this corpus lives mostly in appendix glossaries. That is
  the same FIPS-retrieval weakness the week-2/week-4 retrieval evals measured — the agent
  loop neither fixes nor worsens it, and the proportional budget guaranteed the FIPS
  sub-query still contributed its share of context.
- **Dedupe observed working:** shared chunks across sub-query pairs (e.g. `171r3:appendix-c`)
  appear once in the merged context; the second sub-query still contributed its remaining
  chunks.

## Critic: verdicts recorded, leniency at 8B, fail-open exercised

The critic **passed all five drafts**, including two it should have pushed back on (v1-q23's
draft cites only SP 800-53 chunks while making claims about SP 800-171 scope; v1-q24 answers
a different question than asked). It is not a no-op — an earlier live run on a lookup
question (AC-10) produced two revise verdicts with concrete `uncited_claim` issues, hitting
the revision cap and finalizing with the caveat flag — but as a *self*-critic on its own
drafts, llama3.1:8b is lenient. This is the measured version of a known LLM property
(self-preference), and it is why the benchmark's critic pass-rates must be read per
provider, never pooled.

One trace (v2-q43) recorded `fallback=true`: the critic's JSON never parsed after repair
turns and the loop **failed open to pass** instead of burning revisions — the designed
behavior under ADR-007, now observed live.

## New sentinel misuse variant: mid-answer [NO_ANSWER]

v2-q43's draft answered part 2, then wrote a literal `[NO_ANSWER]` **mid-text** while
refusing part 1 ("For the first part... [NO_ANSWER] The excerpts do not state...") — even
though `03.01.08` was rank 1 in its context. Week 4 documented the *trailing* sentinel
variant (v2-q46), which the agent synthesizer now strips and flags; the mid-text variant is
new, is neither leading nor trailing, and therefore survives into the answer text. Logged as
a known issue: candidates are a prompt rule ("the sentinel may only open the reply") or
post-processing that treats any sentinel occurrence as a partial-refusal signal. Deferred
until cloud traces show whether it reproduces beyond 8B.

## Verification checklist (plan doc)

| Check | Result |
|---|---|
| Plans sensible | 9/10 sub-queries; 1 few-shot anchoring failure (v1-q24, documented) |
| Retrieval per sub-query on-topic | Yes where the corpus states the content; FIPS glossary weakness pre-dates the agent loop |
| Critic verdicts justified | Verdicts recorded per trace; lenient at 8B (2 flawed drafts passed); fail-open observed once |
| Revision cap proven | By test (adversarial fixture, exactly 2 revisions) and live (AC-10 run finalized with caveat) |
| Trace dumps complete | Every node's input/output present for all 5 requests |
