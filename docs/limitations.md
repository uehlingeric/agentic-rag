# Known Limitations

Honest scope statement, last updated for v0.1.0 (2026-07-12). Everything here is
deliberate: this is a reference system that optimizes for measured, reproducible
claims over breadth. Items marked *(candidate)* are tracked as good-first-issue work.

## Corpus and scope

- The corpus is five English-language NIST publications (SP 800-53r5, SP 800-171r3,
  AI RMF 1.0, FIPS 199, FIPS 200). Chunking rules, retrieval settings, the golden
  dataset, and every published number are specific to this corpus. Nothing here
  demonstrates multilingual retrieval, tabular/figure understanding, or web-scale
  ingestion.
- PDF extraction keeps running headers/footers in some chunks; boilerplate
  stripping is a known gap *(candidate)*. Retrieval quality numbers include this
  noise.

## Evaluation caveats

- **LLM-as-judge, not human evaluation.** Scores are judge.v2 rubric scores from a
  cross-judge setup (an answer is never scored by the model that produced it).
  The calibration study ([judge-calibration.md](judge-calibration.md)) reports
  quadratic-weighted kappa per dimension — but its reference labels were
  **model-authored (blind), not human**. Treat absolute scores as consistent
  measurement under a fixed instrument, not ground truth.
- **Judge leniency is real and documented.** The gemini judge runs +0.4–0.9 more
  lenient than the sonnet judge on identical answers. Cross-provider tables are
  only comparable same-judge; the benchmark doc flags every place this matters.
- **Single-run tendencies.** Temperature-0 runs still vary: the same config flipped
  one question between partial-answer and refusal across two runs. Aggregates are
  the unit of comparison, not single questions.
- **Refusal-conversion selection effect.** The agent loop converts some refusals
  into honest partial answers, which then *enter* the judged pool and can lower
  per-dimension means. Agentic-vs-vanilla deltas are unreadable without checking
  refusal flips first ([benchmarks.md](benchmarks.md) does).

## Guardrails

- **Injection screening is a mitigation, not a solve.** The red-team suite passes
  30/30 expect-catch cases with 7 annotated known misses (multilingual, homoglyph,
  leetspeak, split-across-chunks, social-engineering variants). A determined
  attacker gets through the regex layer; the honest claim is defense-in-depth, not
  prevention ([guardrails.md](guardrails.md)).
- **The excerpt-delimiter defense is structural, not behavioral.** The week-6
  canary showed the local 8B model didn't follow embedded injections even
  *without* delimiters — so the measured value of delimiters here is that forged
  excerpt boundaries are impossible (CI-tested), not a demonstrated behavioral
  rescue on a susceptible model.
- **Streaming responses scan the final answer only.** SSE deltas reach the client
  before the output PII scan runs on the complete text (documented in ADR-008/009).
  A deployment that must never emit unscanned tokens has to buffer the stream.
- **Bare 9-digit SSNs and 7-digit phone numbers are not detected** by design
  (false-positive cost); NER-based person/org detection is opt-in and adds spaCy
  inference latency.

## Agent loop

- **The loop helps capable models and hurts small ones.** Cloud models roughly
  halved false refusals under the agent loop; the local 8B got *worse* on every
  rubric dimension with a 17% caveat rate. The loop is not a free upgrade — it
  amplifies planner/critic quality.
- **Relational questions can lose from decomposition.** One benchmark question
  degraded from answer to refusal on both cloud providers because sub-queries
  destroyed the relationship being asked about; keeping a joint sub-query for
  relationship questions is a planner.v2 candidate *(candidate)*.
- **Cost and latency.** The agent loop runs ~2× generation cost and 1.8–2.6× p50
  latency versus vanilla on identical questions.

## Retrieval

- **Local rerankers were a measured negative.** Both the 8B LLM reranker and the
  MiniLM cross-encoder *hurt* NDCG on this corpus, yet the LLM reranker halves
  end-to-end false refusals for cloud providers — kept for that reason, with the
  trade-off published rather than hidden.
- **Context-size mismatch.** Rerank-on configs synthesize from top-8 chunks,
  rerank-off from top-10, so rerank comparisons mix ordering effects with context
  size *(candidate: equalize)*.

## Serving posture

- Static bearer tokens, per-token in-process rate limits, and SQLite/FAISS storage
  are a **development posture**: right-sized for a single-node reference system
  and CI, not a multi-tenant deployment.
- The metrics ledger and audit log are local files; there is no retention policy,
  rotation beyond daily files, or PII-safe export tooling.

## What production would add

Real identity (OIDC/mTLS) and secret management; a shared vector store and
document DB with a corpus-refresh pipeline; buffered streaming behind the output
scan; distributed rate limiting; human-labeled judge calibration and periodic
re-calibration; injection defense-in-depth (canary tokens, tool-call allowlists,
egress controls); autoscaled serving with SLOs and alerting on the exported spans;
and a model-routing layer with fallbacks instead of a per-request provider flag.
