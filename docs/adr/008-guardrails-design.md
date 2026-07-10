# ADR-008: Guardrails — Layered Detectors, Policy-as-Config, Honest Mitigation

**Status:** Accepted — 2026-07-10

## Context

Week 6 adds the governance layer a federal RAG deployment is judged on: PII detection in
and out, prompt-injection screening, a refusal policy, and an audit trail. The design is
constrained less by a feature list than by three commitments the rest of the repo already
made:

- **Honesty over theater.** The project publishes its eval numbers; a guardrail chapter
  that claims 100% injection defense would read as unserious next to them. Heuristics are
  a mitigation, not a solve, and the docs and tests must say so with measured numbers.
- **The quickstart stays keyless and small.** A guardrail that pulls a 500 MB model into
  the default install breaks the "first cited answer in five minutes, no API keys" bar.
- **Benchmarks must stay comparable.** Weeks 4–5 measured the bare pipeline. If guardrails
  silently entered that path, every prior number would shift and the vanilla/agentic delta
  would blur.

## Decision

**Two detector layers, one interface.** PII detection is an always-on regex layer
(SSN, EIN, phone, email, credit card, IPv4) plus an *optional* spaCy NER layer
(PERSON, ORG) behind the `[guardrails-ner]` extra. The regex layer is tuned for a corpus
where `SP 800-53r5`, `AC-2(3)`, and `§3.13.11` must never fire: credit cards are
Luhn-validated, phones require visible NANP structure, SSNs require a consistent
separator. Bare 9-digit runs (SSN) and bare 7-digit runs (phone) are *deliberately not
detected* — indistinguishable from document and section numbers, an honest precision
tradeoff documented in code. NER stays optional because a 12–40 MB model download cannot
sit in the keyless quickstart, and because NER on standards prose is too
false-positive-prone to block or redact on — its entities are flag-only.

**Injection screening is heuristic and openly partial.** Four regex categories
(instruction override, role-play, context escape, encoded payload) screen queries and,
for audit only, retrieved chunks. The red-team suite (`evals/redteam/attacks_v1.jsonl`)
carries annotated *known misses* — multilingual, homoglyph, leetspeak, sentence-split,
soft social-engineering — and a test fails if a documented miss silently starts catching,
so the published catch rate cannot drift from the code. We publish the honest per-category
number, not a round one.

**Retrieved content is data, not instructions.** The corpus-poisoning defense is
structural, not behavioral: `build_context` wraps every excerpt in an
`<excerpt id=n source="...">` delimiter and neutralizes any `<excerpt`/`</excerpt>`
markers inside chunk text, so a poisoned chunk cannot close our delimiter or forge a
higher-authority excerpt. The synthesis prompts (`synthesis.v3`, `agent-synthesis.v2`)
add the matching rule that excerpt content can only be quoted or cited, never followed.
We do not claim the delimiter changes whether a given model obeys an injected
instruction — the live canary (llama3.1:8b) did not follow the injection in any tested
condition, which is defense-in-depth, not a delimiter victory. What the delimiter
*guarantees* is that excerpt boundaries cannot be forged; that guarantee is CI-tested
without an LLM.

**Policy-as-config, defaults in code.** `GuardrailPolicy` maps each entity/direction to
`block | redact | flag`; the conservative federal defaults live in Python (hermetic
tests, no cwd dependence) and `guardrails.yaml` is the deployment override, pinned equal
to the defaults by a test. Unknown keys in a policy file are a hard error (`extra="forbid"`)
so a typo is loud, not silently ignored.

**Guardrails wrap the pipeline from outside.** `GuardedPipeline` is a wrapper, not a
pipeline edit: input scan → inner pipeline → output scan → audit. It is on by default;
`--no-guardrails` and `settings.guardrails.enabled = False` bypass it entirely, and the
eval runner constructs pipelines directly — so every week-4/week-5 benchmark number stays
measured on the bare path. Layering is one-directional: guardrails may import pipeline and
agent modules; neither imports guardrails.

**Every request is audited under a versioned schema.** One `audit_v1` JSONL record per
request (see `docs/audit-log.md`): the query is hashed, not stored, unless
`log_raw_query` is set; detections carry detector/entity/action but never the matched text
or its span (the match may *be* the PII). Fields are append-only; a breaking change bumps
the schema version.

## Consequences

- The injection layer will miss obfuscated and multilingual attacks by construction; the
  honest catch rate and miss list are the deliverable, and the primary containment for a
  missed query-injection is that the corpus is trusted NIST text and excerpts are framed
  as data.
- Guardrail overhead is regex-only on the default path and measured at well under the
  <300 ms p50 bar (see `docs/guardrails.md`); enabling NER adds model-load and inference
  cost and is opt-in.
- The output scanner only guards the final answer, so streamed deltas reach the terminal
  before the output verdict; the CLI surfaces this honestly and agentic streaming is
  refused. A production deployment that must never stream unscanned tokens would buffer —
  a documented, deliberate non-goal for the reference CLI.
