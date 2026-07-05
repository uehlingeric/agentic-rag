# Week 3 Spot-Check: Cited Generation on a Local 8B Model

**Date:** 2026-07-04
**Config:** `agentic-rag ask` pipeline — hybrid retrieval (pool 30) → no rerank → context ≤6000 tokens → `synthesis.v2` → citation validation. Provider: ollama / llama3.1:8b, temperature 0, RTX 5060 Ti 16GB. Paid providers pending API keys; their runs will be appended as a compliance-delta table when available.

Every answer below was reviewed by hand against the golden reference answer AND against the
text of each cited chunk (does the excerpt actually support the claim it is attached to?).

## Prompt version result (the headline)

llama3.1:8b with `synthesis.v1` refused only **2/5** unanswerable questions — it acknowledged
the context lacked the answer, then answered anyway from memory (q26 produced password rules
attributed to SP 800-63, which is not in the corpus). Rewriting the prompt as `synthesis.v2`
(instructions moved *after* the context block, a refusal-first decision procedure, and one-shot
examples of a refusal and a cited answer) fixed refusals completely and improved citation
presence, measured over 15 questions on identical retrieved contexts:

| Prompt | Unanswerable refused | Lookup answered with ≥1 citation |
|---|---|---|
| synthesis.v1 | 2/5 | 7/10 |
| synthesis.v2 | **5/5** | **9/10** |

Instruction position and worked examples mattered; rule *wording* alone did not (an
intermediate v2 draft with stricter rules but instructions still before the context measured
2/5 and 6/10 — worse than v1 on citations).

## Refusal behavior: 5/5

All five unanswerable golden questions (q26–q30) produced the `[NO_ANSWER]` sentinel with a
sensible one-sentence explanation, e.g. q28: *"The excerpts do not mention an estimated cost
range for implementing a comprehensive security control baseline per FIPS 200."* No citations,
no hallucinated content, `Answer.refusal=True` end to end. Exit criterion met on ollama.

## Ten answerable questions

Citation validity = every `[n]` marker resolves to a context chunk. Attribution = the cited
excerpt actually supports the claim it is attached to (judged by hand).

| Q | Type | Answer correctness | Markers valid | Attribution |
|---|---|---|---|---|
| q01 | lookup | Correct (account types) | 1/1 | Weak — quotes AC-2 language but cites a 171 crosswalk table |
| q04 | lookup | **Wrong** — answers about emergency accounts, not 3.1.1 | 1/1 | Faithful to its excerpt, excerpt off-topic |
| q07 | lookup | **Wrong** — names a different IA-5 item | 0 markers | Uncited (the one citation-presence failure) |
| q11 | synthesis | Mostly correct AC-1/AC-2/AC-10 synthesis | 3/3 | **All three markers rotated** — policy claims cite the AC-10 chunk, session claims cite a tailoring table |
| q12 | synthesis | Correct | 2/2 | Correct |
| q14 | synthesis | Correct | 4/4 | Correct (incl. a grounded PE-4 aside) |
| q16 | synthesis | Partial — mapping muddled, relationship not really explained | 3/3 | Mixed |
| q21 | multihop | **Wrong** — misstates the confidentiality definition | 2/2 | Resolve but weakly support |
| q23 | multihop | Mostly correct compare/contrast, one overclaim | 3/4 | Mixed; invalid `[9]` stripped and reported by the validator |
| q25 | multihop | Correct | 3/3 | Correct |

Tally: 5/10 correct or mostly correct, 2 partial, 3 wrong. 9/10 carried at least one valid
citation; exactly one invalid marker appeared across all runs and the post-processor stripped
and reported it as designed.

## Findings

1. **Marker validity ≠ attribution.** The mechanical guarantee (every surviving `[n]` resolves
   to a real chunk) held everywhere. But in about half the answers at least one marker points
   at an excerpt that does not support its claim — q11 is the extreme case, with content largely
   correct and every marker attached to the wrong excerpt. Nothing in the current pipeline can
   see this. This motivates a groundedness eval (claim ↔ cited-chunk entailment) as a week-4/5
   candidate, and is the metric to watch when comparing paid providers.
2. **Answer quality is model-bound, not retrieval-bound.** In 8/10 cases the golden chunk was
   present in the context the model saw; errors were misreading or mis-selection (q04 picked
   the wrong excerpt to paraphrase; q21 pulled the wrong definition). A stronger synthesis
   model should lift correctness without touching retrieval.
3. **Refusal compliance is a prompt-structure problem before it is a model problem.** Same
   model, same contexts: 2/5 → 5/5 purely from prompt layout (see table above).
4. **Latency** (RTX 5060 Ti): retrieve 0.02–0.09s; synthesis 0.5–4.8s scaling with answer
   length; refusals are fast (~0.6s, short outputs). Rerank adds ~4s/query with the local 8B
   as listwise judge (see benchmark doc).

## Caveats

- Single run per question at temperature 0; ollama greedy decoding was reproducible across
  the runs above but was not stress-tested for cross-version drift.
- Judgments are one reviewer's reading of the golden references and cited excerpts.
- fips-199/fips-200 questions ground in glossary/appendix chunks only (ADR-003 corpus quirk);
  q25's correct answer cites the fips-200 glossary rather than body text, as expected.
