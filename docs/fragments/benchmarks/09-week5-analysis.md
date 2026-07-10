### Analysis

**Where agents pay for themselves: refusal conversion and multi-hop, at ~2× cost.** For
both cloud models the agent loop nearly halved false refusals (anthropic 6 → 3 of 38
answerable questions net, google 7 → 4): per-sub-query retrieval assembles evidence that
one-shot retrieval missed, so questions the vanilla pipeline refused come back as grounded
answers. On the multi-hop slice under the same (sonnet) judge, google improved on
faithfulness 4.60 → 4.88 and citation accuracy 4.60 → 5.00; the lookup slice is flat for
both cloud providers, which is the planner doing its job — 52–61% of questions passed
through undecomposed, so simple questions never paid the agent tax. The tax where it is
paid: generation cost roughly doubles (anthropic $1.31 → $2.74, google $0.66 → $1.25 per
46-question run) and p50 latency multiplies 1.8–2.6× (google 4.4 → 7.7 s, anthropic
15.8 → 41.0 s). That lands at the low end of the "2–4× cost" hypothesis in the week-5 plan.

**Read scoring composition before reading deltas.** Refusals are never judged, so when the
agentic pipeline converts a refusal into an answer, that answer *enters* the judged pool —
usually as a partial, honestly-caveated response. Anthropic's multi-hop relevance
"drop" (5.00 → 4.56) is entirely this effect: the two sub-5 rows (v1-q25, v2-q44) are
exactly the two questions vanilla refused outright, scored R=3 for answering the part the
context supported and saying so. An answered-with-caveats partial beats "not found in
corpus" for a user, but it scores worse than the refusal it replaced, which is invisible to
the mean. The same caveat applies to anthropic's flat 5.00s (gemini-judge leniency,
week 4): within-provider deltas are meaningful; cross-provider rows are not a ranking.

**Decomposition can destroy a relational question.** v1-q16 ("how are 3.1.1 and 3.1.2
related?") flipped the other way — answered by vanilla (F=5), refused by agentic — on
*both* cloud providers, with near-identical plans: the planner split it into one sub-query
per requirement, each retrieved its own requirement's text cleanly, and the synthesizer
correctly found no chunk stating the *relationship* — the joint framing the single-query
pipeline had preserved. Planner.v2 candidate: relationship questions should keep a joint
sub-query alongside the per-entity ones.

**An 8B model cannot drive this loop.** ollama/llama3.1:8b got worse on every rubric under
the agent (faithfulness 2.97 → 2.51, citation accuracy 3.08 → 2.41), hit the revision cap
with unresolved critique on 17% of questions (cloud: 0%), and its false-refusal "win"
(0.08 → 0.03) is the bad kind — answering even more often from weak context. Each extra
loop stage (plan, critique, revise) is another LLM call that can go wrong, and at 8B they
do: few-shot anchoring in plans and a self-lenient critic were both observed directly in
the trace review (`docs/reviews/week-05-traces.md`). The loop amplifies the model driving
it, in both directions.

**Ops.** Mean revisions stayed near zero on cloud models (anthropic 0.13, google 0.02) —
the critic gates rather than iterates when the draft is already grounded. Judge costs ran
$0.16–0.40 per config. Full provenance: dataset v2 (46 eval items), planner.v1 /
agent-synthesis.v1 / agent-revise.v1 / critic.v1, judge.v2, per-row judge identity in
`evals/results/generation-20260710-131027Z/`.
