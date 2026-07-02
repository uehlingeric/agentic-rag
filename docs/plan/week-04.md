# Week 4 — Eval Harness + First Full Benchmark

**Dates:** Mon Jul 27 – Sun Aug 2, 2026
**Objective:** The differentiator week. A generation eval harness with LLM-as-judge scoring — calibrated against human labels — and the first full benchmark matrix (provider × retrieval config) committed to the README-bound benchmarks doc. By Sunday, `make eval` runs the entire suite and emits a markdown report.

## Exit Criteria

- [ ] Golden dataset expanded to 50 (30 existing + 20 new, same authoring standard)
- [ ] Judge scores faithfulness, answer relevance, and citation accuracy on 1-5 rubrics
- [ ] Judge calibration: ≥0.6 weighted Cohen's kappa vs human labels on a 20-item subset; calibration doc committed
- [ ] Full matrix run: ≥3 providers × {bm25, dense, hybrid, hybrid+rerank} with quality, latency, and cost columns
- [ ] `docs/benchmarks.md` is regenerated from results by script, never hand-edited

## Workstreams

### 1. Judge design
- [ ] Rubrics in versioned prompt files: faithfulness (claims supported by cited chunks), relevance (answers the question asked), citation accuracy (citations point to supporting text)
- [ ] Judge receives question + answer + cited chunk texts — never the reference answer for faithfulness (grounding is against retrieved context)
- [ ] Structured output: score + one-line justification per dimension; strict JSON with retry-on-parse-failure
- [ ] Judge model configurable; default to strongest available; judge provider ≠ generation provider rule to reduce self-preference bias, documented in ADR-006

### 2. Human calibration
- [ ] Hand-label 20 stratified items (all question types) on all three rubrics — labels in `evals/calibration/human-labels.jsonl`
- [ ] Agreement analysis: weighted kappa per dimension; iterate rubric wording until ≥0.6 or document irreducible disagreement
- [ ] `docs/judge-calibration.md`: method, iterations, final agreement, known judge failure modes

### 3. Eval runner
- [ ] `evals/run_generation.py`: config matrix → per-item results JSONL (answer, scores, tokens, cost, latency) → aggregate table
- [ ] Concurrency with rate-limit awareness; resumable on partial failure; cost estimate printed before live runs, `--confirm` gate above $5
- [ ] Report generator: `docs/benchmarks.md` from results dir (tables + config/versions/dates); refusal correctness on unanswerable items reported separately
- [ ] `make eval` = retrieval evals + generation evals + report

### 4. Dataset expansion
- [ ] +20 questions targeting gaps from the coverage matrix (AI RMF and FIPS underrepresented; more multi-hop)
- [ ] Dataset versioned `v2`; results always record dataset version

## Verification

- Judge unit tests with fixture answers engineered to score high/low per dimension (a wrong-citation answer must fail citation accuracy).
- Two consecutive full runs on one config: aggregate scores within ±0.2 (judge variance check; if worse, add per-item score averaging over 3 judge calls).
- Kappa ≥ 0.6 on all three dimensions.

## Commit Milestones (4-6 commits)

1. Judge rubrics + structured scoring + tests
2. Calibration labels + agreement analysis + doc
3. Eval runner with resume + cost gate
4. Dataset v2
5. First full benchmark + generated report

## Risks & Notes

- **Budget:** full matrix ≈ 50 questions × 12 configs ≈ 600 generations + 1,800 judge calls. Estimate before running; use mid-tier generation models where the matrix explodes, flagship judge only.
- Calibration is the credibility of every number published at launch — do not skip iteration to hit the week.
- If a provider is systematically judged worse, verify with human spot-checks before publishing; add a caveat note if bias is suspected.
