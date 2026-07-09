# ADR-006: LLM-as-Judge Design — Cross-Provider Assignment, Single-Call Rubrics, Calibration Before Trust

**Status:** Accepted — 2026-07-09

## Context

Week 4 introduces generation quality scores (faithfulness, answer relevance, citation
accuracy) that will headline the published benchmarks. Retrieval metrics are computable
from labels; generation quality is not — an LLM judge scores free-text answers, which
imports three risks the design must manage rather than ignore:

- **Self-preference bias.** Models measurably prefer their own outputs. A benchmark
  where claude judges claude and gemini judges gemini is comparing self-images, not
  systems.
- **Ungrounded scoring.** A judge shown the reference answer scores string similarity;
  a judge free to use its own knowledge scores world-model agreement. Neither is
  faithfulness — the property we want is "supported by the excerpts the answer cited."
- **Uncalibrated numbers.** A 4.2/5 means nothing unless judge scores are shown to
  track an independent labeler's on the same rubric.

## Decision

**Cross-provider assignment.** An answer is never scored by the provider that generated
it. `Settings.judge.providers` is a preference list (default `["anthropic", "google"]`);
the runner assigns the first entry that differs from the generation provider. With the
week-4 matrix that means the strongest available judge (claude-sonnet-4-6 via Bedrock)
scores ollama and gemini answers, and gemini-3.5-flash (via Vertex) scores claude
answers. The cost of the rule is that headline rows are not all scored by the same
judge; a dual-judged overlap slice (one full config scored by both judges) is run with
every published matrix to quantify inter-judge agreement, and per-row judge identity is
recorded so no comparison is ever anonymous.

**Grounding inputs, not references.** The judge receives exactly: the question, the
answer text with its inline `[n]` markers, and the text of the chunks those markers
cite — never the golden reference answer, and never the un-cited context. Faithfulness
and citation accuracy are therefore properties of the answer-evidence pair, judged
identically for every provider. Refusals are not rubric-scored at all: refusal
correctness on unanswerable items is a separate count, because averaging "declined to
answer" into a 1–5 faithfulness scale rewards cowardice on answerable questions and
hides it on unanswerable ones.

**One call, three dimensions, strict JSON.** A single judge call scores all three
rubrics (versioned `judge` prompt, temperature 0; `judge.v2` is the first version to
pass calibration for both judges — see `docs/judge-calibration.md`), returning a bare JSON object with an
integer 1–5 score and a one-sentence justification per dimension. Malformed replies get
bounded conversational repair (the bad reply plus a corrective instruction are
appended, `max_parse_retries` times) before the item is recorded as a judge failure —
never silently dropped, never guessed. One call per item keeps the three scores
mutually consistent, costs a third of per-dimension calls, and the rubric text
explicitly instructs independent scoring across dimensions.

**Calibration gates publication.** Before matrix scores are published, the judge is
scored against independent labels on a 20-item stratified subset; quadratic-weighted
Cohen's kappa ≥ 0.6 per dimension is the bar, and rubric wording iterates as new prompt
versions (per ADR-005) until it is met or the irreducible disagreement is documented.
Week 4's labels are authored by the coordinator model (Claude Fable), which is distinct
from both judge models but is not a human — `docs/judge-calibration.md` states this
provenance plainly, and replacing these with human labels is standing work, not a
closed question.

Alternatives rejected: a single shared judge for all providers (cleanest comparability,
but the judge's sibling model appears in the matrix and its rows would carry
self-preference; rejected while the matrix is only three providers, revisit if OpenAI
lands); per-dimension judge calls (3× cost for no observed parsing benefit); judging
against reference answers (measures paraphrase distance, not grounding); provider-native
structured-output APIs (ties the judge to per-provider features; the repair loop is
provider-agnostic and its retry count is observable in usage).

## Consequences

- Every result row records judge provider, judge model, and judge prompt id; benchmark
  tables state which judge scored which rows. No number is separable from its judge.
- The judge prompt is a versioned contract (`judge.v1.md`): calibration pins the
  version it validated, and any wording change re-enters calibration before its scores
  publish.
- Judge cost is a first-class benchmark column: cross-provider assignment means paid
  judge calls even for free (ollama) generations, and the runner's cost estimate and
  `--confirm` gate cover judging, not just generation.
- If the two judges disagree systematically on the overlap slice, that is a published
  finding with a caveat note, not a silent averaging decision — the week-4 plan's
  bias-verification rule (human spot-check before publishing a provider-adverse result)
  applies. Week 4's overlap run found exactly this: gemini-3.5-flash scores identical
  answers +0.4 to +0.9 higher than claude-sonnet-4-6 (leniency, not self-preference — it
  is equally lenient on its own answers), so the benchmark carries a same-judge
  comparison alongside the cross-judge headline table.
