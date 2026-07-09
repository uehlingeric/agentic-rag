# Judge Calibration — Week 4

**Judges:** `claude-sonnet-4-6` (Bedrock, `global.` profile) and `gemini-3.5-flash` (Vertex,
`global` endpoint), assigned per ADR-006 (an answer is never scored by the provider that
generated it). **Rubric:** `judge.v1` → `judge.v2` (one iteration, below). **Gate:**
quadratic-weighted Cohen's kappa ≥ 0.6 against independent labels, per dimension, per judge.

## Label provenance — read this first

The reference labels were **not produced by a human**. They were authored by the build's
coordinator model (Claude Fable), which is a distinct model from both judges, working from a
blind labeling sheet (question + answer + cited excerpts only — no judge scores, no reference
answers) and scoring against the same `judge.v1` rubric anchors. The labels file records this
in every row's `labeler` field. This calibrates judge-vs-independent-careful-reader agreement;
it does not certify human agreement. Replacing these labels with human labels is standing
work, and every kappa in this document should be read with that caveat.

## Method

1. A stratified 20-item set was selected deterministically (seed 13) from the full 12-config
   matrix run: unique questions only, refusal rows excluded (they carry no rubric signal),
   spread across providers (11 ollama / 5 google / 4 anthropic), retrieval modes, and all four
   question types — including three unanswerable questions that were (incorrectly or
   defensibly) answered, which anchor the low end of the faithfulness scale.
   Selection: `evals/run_calibration.py sheet`; artifacts in `evals/calibration/`.
2. The coordinator labeled all 20 items on the three 1–5 rubrics
   (`evals/calibration/labels.jsonl`), reading every cited excerpt in full.
3. Both judges re-scored the identical 20 rows (`evals/rejudge.py`), so each judge has a
   complete labels-vs-judge comparison, including answers from its own provider.
4. Agreement: `evals/run_calibration.py agreement` (quadratic-weighted kappa, exact match,
   mean absolute difference, |diff| ≥ 2 audit trail).

## Round 1 — judge.v1

| Judge | Faithfulness κ | Relevance κ | Citation accuracy κ |
|-------|----------------|-------------|---------------------|
| claude-sonnet-4-6 | 0.934 | 0.821 | 0.750 |
| gemini-3.5-flash | 0.782 | **0.343** | 0.782 |

The sonnet judge passed all three dimensions. Gemini failed relevance badly. The |diff| ≥ 2
audit showed every gemini relevance miss was the same failure mode: **saturation on partial
answers** — answers that fully addressed only one half of a two-part question (v1-q14,
v1-q25, v2-q46) received a 5 where the rubric's own anchor text says 3. This mirrors the
matrix-level observation (below) that gemini compresses scores upward.

## Iteration — judge.v2

One targeted change to the relevance rubric (per ADR-005, a new immutable file
`judge.v2.md`; v1 is untouched): the 5-anchor now says "every part of the specific
question", the 3-anchor explicitly covers "answers only part of a multi-part question …
even when the omission is acknowledged honestly", and a closing rule states that an answer
addressing one of two parts scores at most 3. Faithfulness and citation anchors are
byte-identical to v1.

## Round 2 — judge.v2 (published configuration)

| Judge | Faithfulness κ | Relevance κ | Citation accuracy κ |
|-------|----------------|-------------|---------------------|
| claude-sonnet-4-6 | 0.934 | 0.680 | 0.805 |
| gemini-3.5-flash | 0.782 | 0.737 | 0.782 |

All six judge × dimension cells clear the 0.6 gate. Gemini's relevance recovered
(0.343 → 0.737). The sonnet judge's relevance dipped (0.821 → 0.680) — the multi-part cap
makes it slightly harsher than the labeler on two honest-partial answers — but remains
comfortably above the bar. The full benchmark matrix is therefore scored with **judge.v2**,
and `docs/benchmarks.md` reports v2 numbers exclusively.

## Judge variance (same judge, same rows, two runs)

Two independent sonnet-judge runs over the 41 judged ollama/hybrid answers (temperature 0,
`judge.v1`): per-dimension aggregate deltas were ≤ **0.024** on the 1–5 scale against the
±0.2 gate. Single-call judging is stable enough; per-item averaging over 3 calls is not
needed.

## Judge severity and self-preference (overlap slices)

The same 41 ollama/hybrid answers scored by both judges (`judge.v1`):

| Judge | Faithfulness | Relevance | Citation accuracy |
|-------|--------------|-----------|-------------------|
| claude-sonnet-4-6 | 2.98 | 3.51 | 3.00 |
| gemini-3.5-flash | 3.39 | 4.37 | 3.29 |

Gemini scores the identical answers +0.4 to +0.9 higher — a systematic severity gap, largest
on relevance. Self-judging slices resolve the direction of the bias: gemini awarded flat
5.00s **both** to claude's answers (cross-judge) and to its own gemini answers (self-judge),
while the sonnet judge discriminated even against its own provider's answers (self-judged
anthropic/hybrid: F 4.48, R 4.90, C 4.61 — *lower* than gemini's cross-judge 5.00s). The
anomaly is leniency/score-compression in gemini, not self-preference in either judge.

**Publication consequence:** cross-provider judge assignment means the anthropic rows in the
benchmark are scored by the more lenient judge. Under a single judge (sonnet, `judge.v1`
overlap), google answers scored slightly *higher* faithfulness than anthropic's own
(4.69 vs 4.48) while anthropic led on relevance (4.90 vs 4.62) — i.e., the apparent
anthropic sweep in the headline table is partly a judge-assignment artifact. The benchmark
doc carries this caveat next to the table.

## Known judge failure modes

- **Score compression (gemini):** clusters at 5 for any fluent, mostly-grounded answer;
  `judge.v2`'s explicit multi-part rule recovers discrimination on relevance, but per-row
  gemini scores remain right-shifted relative to the sonnet judge on identical answers.
- **Honest-partial penalty ambiguity:** answers that answer half a question and correctly
  state the excerpts lack the rest (v2-q33, v2-q46) sit between rubric anchors; v2 pins them
  at ≤ 3 relevance by rule, which the sonnet judge sometimes applies more harshly than the
  labeler (its post-iteration relevance dip).
- **Citation edge cases:** answers with no inline markers but a trailing bare reference line
  (v1-q26), and stripped-invalid-marker answers (v1-q27), produce the widest citation-accuracy
  disagreements (|diff| ≥ 2) — the rubric under-specifies "markers absent where required"
  versus "reference present in nonstandard form".
- **Sentinel misuse observed in generation (not judging):** one anthropic answer emitted a
  *trailing* `[NO_ANSWER]` after a partial answer (v2-q46); the pipeline only strips the
  leading sentinel, so this survives into judged text. Recorded as a week-5 candidate fix.

## Reproduction

Artifacts: `evals/calibration/` (sheet, selected rows, labels, per-judge/per-version rejudge
files). Judge-version provenance for every score is in each row's `judge.prompt_id`. The v1
matrix judge blocks are archived under
`evals/results/generation-20260709-200512Z/judge-v1-archive/`.
