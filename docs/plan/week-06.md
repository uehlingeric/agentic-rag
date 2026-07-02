# Week 6 — Guardrails, Refusal Policy, Audit Logging

**Dates:** Mon Aug 10 – Sun Aug 16, 2026
**Objective:** The governance layer that mirrors production federal deployments: input/output PII detection, prompt-injection screening, a configurable refusal policy, and structured audit logging. By Sunday the pipeline runs with guardrails on by default, a red-team test suite passes, and every request leaves an audit trail.

## Exit Criteria

- [ ] PII in queries and answers is detected and handled per policy (block, redact, or flag — configurable)
- [ ] Prompt-injection attempts in the red-team suite are caught or neutralized (document detection rate honestly)
- [ ] Every request writes a structured audit record; audit log documented and schema-versioned
- [ ] Guardrails add <300ms p50 overhead on the local path (measured, in benchmarks doc)
- [ ] Eval suite re-run with guardrails on: zero degradation on clean golden questions (false-positive check)

## Workstreams

### 1. PII detection
- [ ] Input scanner: regex layer (SSN, EIN, phone, email, credit card, IP) + spaCy NER layer (PERSON, ORG in sensitive patterns) — behind one `PIIScanner` interface
- [ ] Output scanner: same detectors on generated answers pre-return
- [ ] Policy actions per entity type in `guardrails.yaml`: `block | redact | flag` with defaults mirroring a conservative federal posture (block on input SSN, redact output emails, etc.)
- [ ] Test suite: fixture corpus of true positives, tricky formats (spaced SSNs, obfuscated emails), and hard negatives (NIST control ids that look like ids, phone-like section numbers)

### 2. Prompt-injection screening
- [ ] Heuristic layer: instruction-override patterns, role-play jailbreak markers, context-escape attempts, encoded payloads (base64/hex heuristics)
- [ ] Retrieved-content defense: chunks are data, not instructions — delimiter discipline in synthesis prompt + canary-token test proving injected instructions in a poisoned chunk are not followed
- [ ] Red-team suite: ≥30 attack cases across query-injection and corpus-poisoning; results table with per-category catch rate (publish the honest number; 100% claims read as unserious)

### 3. Refusal policy
- [ ] Policy config: out-of-corpus questions → grounded refusal (already working — formalize); blocked-input template; PII-block explanation template
- [ ] Refusal responses carry machine-readable `refusal_reason` in JSON output (downstream systems need this in real deployments)

### 4. Audit logging
- [ ] JSONL audit record per request: timestamp, request id, query hash (not raw query by default — privacy), provider/model, pipeline config, guardrail verdicts, chunk ids retrieved, usage, latency per stage, refusal reason if any
- [ ] Schema versioned (`audit_v1`); rotation-friendly file layout; `docs/audit-log.md` with field reference and a worked example
- [ ] Config flag for raw-query logging (default off) — mirrors real deployment debates; document the tradeoff

### 5. Wire-through
- [ ] Guardrails as pipeline stages (input pre-planner, output post-critic) in both vanilla and agentic paths, on by default, `--no-guardrails` escape hatch for benchmarking
- [ ] Overhead measurement added to benchmark tables

## Verification

- Full red-team suite in CI (no live LLM needed for heuristic layers; canary test uses stub provider playback).
- Clean-question false-positive run: 50 golden questions with guardrails on → 0 incorrect blocks.
- Audit log replay: reconstruct one full request lifecycle from the log alone — documented walkthrough in `docs/audit-log.md`.

## Commit Milestones (4-6 commits)

1. PIIScanner + fixture suite
2. Injection heuristics + canary/poisoning tests
3. Refusal policy + templates
4. Audit logging + schema doc
5. Pipeline wire-through + overhead benchmark

## Risks & Notes

- Do not overclaim: heuristic injection defense is a mitigation, not a solve. The README language at launch must reflect that nuance — it reads as expertise, not weakness.
- spaCy model size affects the quickstart footprint — make NER layer optional (`[guardrails-ner]` extra), regex layer always on.
- This week is the strongest interview-story material in the repo (safety sandwich, HITL posture, audit trails) — keep the docs narrative-quality.
