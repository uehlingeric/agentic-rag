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

---

# Cloud Traces: Same Five Questions on sonnet-4-6 and gemini-3.5-flash

**Date:** 2026-07-10, accompanying benchmark run `generation-20260710-131027Z`.
**Config:** identical to the ollama runs above; providers claude-sonnet-4-6 (Bedrock
`global.` profile) and gemini-3.5-flash (Vertex, global endpoint). 10 traced requests,
$0.82 total. Each trace reviewed by hand as above.

## Headline

All three deferred 8B findings are now resolved, one in the unexpected direction. Few-shot
anchoring: **8B-only** — neither cloud planner copied AC-2 into v1-q24. Critic leniency:
**8B-only** — sonnet's critic revised 3 of 5 initial drafts with substantive, typed
citation-grounding issues. Mid-answer `[NO_ANSWER]`: **not 8B-only** — it reproduced on
both cloud providers in the benchmark rows, including once in the *vanilla* pipeline, so
the fix belongs in shared synthesis post-processing (week 6), not the agent path.

## Plans: 10/10 well-formed, zero parse fallbacks

Both providers classified all five questions `multi_hop` with two sub-queries each. Sonnet's
sub-queries adapt to the question — v1-q24 became *"SP 800-53 Revision 5 security controls
for protecting AI system resources such as SR and SA control families"*, exactly the
generalization llama3.1:8b failed to make; planner.v1's few-shot anchoring is an 8B
capability floor, not a prompt defect, so planner.v2's few-shot diversification is
deprioritized. Gemini's sub-queries are terser and once under-specified: its v1-q24
sub-query 2 dropped the AI framing entirely (*"...protecting system and information
resources"*), retrieval came back generic, and the synthesizer refused — consistent with
gemini refusing v1-q24 in both benchmark configs.

## Critic: engaged at cloud scale, and a new behavior — revision to refusal

Sonnet's critic issued `revise` on 3 of 5 initial drafts, every issue a concrete grounding
defect (`unsupported_citation`, `uncited_claim`, `incomplete`), with monotonic convergence:
v1-q23 went 2 issues → 1 → pass across exactly the bounded revisions; v1-q24 cleared its
two issues in one revision. v1-q25 is the interesting one: after two substantive revise
verdicts the third synthesis produced a *refusal* draft — under critic pressure the model
concluded its context (the known FIPS-glossary retrieval gap) could not ground the asked-for
relationship and chose honest refusal over an unsupported answer; the critic is then
skipped by design (`refusal draft`). Gemini's critic passed every answered draft with zero
revisions, matching its benchmark mean of 0.02 — critic pass-rates remain per-provider
reads, never pooled.

## Sentinel misuse: cross-provider, cross-pipeline, always partial answers

The traces themselves are sentinel-clean, but the benchmark rows are not: sonnet (agentic,
v2-q44) and gemini (**vanilla**, v1-q18) both emitted mid-answer `[NO_ANSWER]` in exactly
the ollama v2-q43 pattern — answer the supported part, emit the sentinel, explain the
unsupported part. It appears precisely on partial-answer items, which suggests treating any
non-leading sentinel occurrence as a partial-refusal signal and stripping it in shared
post-processing. Promoted from "deferred pending cloud reproduction" to a week-6 work item.

## Run-to-run variance caveat

anthropic v1-q25 was an honestly-caveated partial answer in the benchmark run but a refusal
in this trace run — same config, same context budget. Single-question refusal behavior is a
tendency, not a constant; the benchmark's aggregate refusal rates are the meaningful unit.

## Verification checklist (cloud)

| Check | Result |
|---|---|
| Plans sensible | 10/10; no parse fallbacks; no few-shot anchoring on either provider |
| Critic verdicts justified | sonnet: 3/5 revised, all issues substantive, 1 revision-to-refusal; gemini: 0 revisions (per-provider read) |
| Sentinel variant reproduced? | Yes — benchmark rows v2-q44 (sonnet agentic), v1-q18 (gemini vanilla); week-6 fix in shared post-processing |
| Revision cap respected | Max 2 revisions observed (v1-q23, v1-q25); no caveat flags |
| Trace dumps complete | Every node's input/output present for all 10 requests |
