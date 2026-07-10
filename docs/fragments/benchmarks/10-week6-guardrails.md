## Week 6 — Guardrails: overhead and false-positive check (2026-07-10)

Guardrails (PII detection, prompt-injection screening, refusal policy, audit logging) are on
by default and wrap the pipeline from the outside ([ADR-008](adr/008-guardrails-design.md),
[docs/guardrails.md](guardrails.md)); the benchmark tables above are unaffected because the
eval runner bypasses the wrapper. Two numbers matter for the benchmark story — that the
governance layer is cheap, and that it does not degrade clean questions
(`evals/results/guardrails-20260710-verify/`, reproduce with `make verify-guardrails`):

- **Overhead.** The full input+output scan+policy path, measured over 210 real
  question/answer pairs from the week-5 run, adds **p50 0.16 ms, p95 0.55 ms** — the default
  path is regex-only, three orders of magnitude under the 300 ms p50 bar. The optional spaCy
  NER layer adds inference cost and is opt-in (`[guardrails-ner]`).
- **No clean-question degradation.** With guardrails on, **0 of 50** golden questions were
  blocked at input and **0 of 210** non-refusal benchmark answers were blocked at output — no
  false-positive refusals introduced.

The injection layer's honest per-category catch rate (30/30 expect-catch cases, 7 documented
misses) and the corpus-poisoning canary live in [docs/guardrails.md](guardrails.md); the
red-team suite is `evals/redteam/attacks_v1.jsonl`.
