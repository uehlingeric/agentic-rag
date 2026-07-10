# Audit Log

Every request through `GuardedPipeline` writes exactly one structured record. The audit log
is the deployment-facing artifact a federal reviewer asks for first: what was asked, what
the guardrails decided, what the model was given, what it cost, and how long it took —
without storing the query text by default.

- **Location:** `{data_dir}/audit/audit-YYYYMMDD.jsonl` (override with
  `guardrails.audit_dir`; disable with `guardrails.audit_enabled: false`).
- **Rotation:** one file per UTC day, keyed by the record's own timestamp (so replay and
  tests are deterministic).
- **Format:** JSON Lines — one self-describing record per line, appended.
- **Schema:** versioned `audit_v1`. Fields are append-only; a breaking change bumps the
  version so downstream consumers can route on `schema`.

## Privacy posture

- **The raw query is not stored by default.** Each record carries `query_sha256`, a
  SHA-256 of the original question. Set `guardrails.log_raw_query: true` to also store
  `raw_query` — this mirrors the real deployment debate (debuggability vs. minimizing
  retained sensitive input) and is off by default. When off, `raw_query` is `null`.
- **Detections never carry the matched text or its span.** A detection records only
  `detector`, `entity`, and `action`. The matched substring may *be* the PII, and a span is
  useless without the text — so neither is written. The audit answers "an email was
  redacted from the output," never "which email."
- **The answer is hashed, not stored.** `answer_sha256` lets you prove which answer was
  returned (e.g. against a cached copy) without retaining it; it is `null` when the request
  was blocked at input and no answer exists.

## Field reference (`audit_v1`)

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | `"audit_v1"` — always first. |
| `request_id` | string | Unique per request (uuid4 hex). |
| `ts` | string | UTC ISO 8601 timestamp; drives daily rotation. |
| `query_sha256` | string | SHA-256 of the original question. |
| `raw_query` | string \| null | Original question, only if `log_raw_query`; else `null`. |
| `provider` / `model` | string | Generation provider and resolved model id. |
| `pipeline` | string | `"vanilla"` or `"agentic"`. |
| `mode` / `rerank` | string | Retrieval mode and rerank stage. |
| `guardrails_enabled` | bool | Always `true` in guarded records. |
| `policy_version` | int | `GuardrailPolicy.version` in force. |
| `ner` | bool | Whether the spaCy NER layer was active. |
| `input_scan` | object \| null | `{detections: [...], blocked}` for the query scan. |
| `output_scan` | object \| null | Same for the answer scan; `null` if input was blocked. |
| `retrieved_flagged_chunk_ids` | string[] | Chunk ids that tripped the injection scanner. |
| `chunk_ids` | string[] | Context chunk ids shown to the model. |
| `input_tokens` / `output_tokens` | int | Token usage summed across the request. |
| `cost_usd` | float \| null | Estimated spend; `null` when unpriceable, `0.0` local. |
| `latency_s` | object | Per-stage seconds incl. `guardrails_in`/`guardrails_out` + `total`. |
| `refusal` | bool | Whether the returned answer was a refusal. |
| `refusal_reason` | string \| null | `out_of_corpus` \| `input_pii` \| `input_injection` \| `output_pii` \| `null`. |
| `answer_sha256` | string \| null | SHA-256 of the returned answer text; `null` when blocked at input. |

A detection entry inside `input_scan`/`output_scan`:

```json
{ "detector": "regex", "entity": "email", "action": "redact" }
```

`detector` is `regex` | `ner` | `injection`; `entity` is a PII entity or injection category;
`action` is `block` | `redact` | `flag`.

## Worked example: reconstructing a request from the log alone

A single clean request (`log_raw_query` on for illustration):

```json
{
  "schema": "audit_v1",
  "request_id": "02c4721484c94f8681443484f42d034f",
  "ts": "2026-07-10T19:53:13.775600+00:00",
  "query_sha256": "2dde4eaa259113d68f1eefa169de081fc09d7a9db0f511a1b90105f4b457182f",
  "raw_query": "What does control AC-2 require organizations to define and document?",
  "provider": "ollama",
  "model": "llama3.1:8b",
  "pipeline": "vanilla",
  "mode": "hybrid",
  "rerank": "none",
  "guardrails_enabled": true,
  "policy_version": 1,
  "ner": false,
  "input_scan": { "detections": [], "blocked": false },
  "output_scan": { "detections": [], "blocked": false },
  "retrieved_flagged_chunk_ids": [],
  "chunk_ids": [
    "7d896e3761f4ea38", "84c729d96620e573", "07d36d4ca6df05e3", "00c69d175f81c886",
    "c6cda2a1e08159ac", "2d0017041b7539e5", "d1ee7c835dd7e081", "3da4f9cad625f5eb"
  ],
  "input_tokens": 3359,
  "output_tokens": 40,
  "cost_usd": 0.0,
  "latency_s": {
    "retrieve": 0.338, "rerank": 0.00002, "synthesize": 1.576,
    "guardrails_in": 0.00004, "guardrails_out": 0.00004, "total": 1.914
  },
  "refusal": false,
  "refusal_reason": null,
  "answer_sha256": "aa45c811c2e9c43f6155a4102692e0f10dbb3d85c4ed70f1933e61dd27c8c51d"
}
```

Read it as a lifecycle:

1. **Ingress.** Request `02c4721484…` arrived at `19:53:13Z`. The query hashes to
   `2dde4eaa…`; because `log_raw_query` was on here, we can also see the question verbatim
   (in the default posture we would have only the hash).
2. **Input guardrail.** `input_scan.blocked = false` with no detections — the query carried
   no PII or injection markers, so nothing was redacted and the pipeline ran on the original
   text. Cost of this stage: `guardrails_in = 0.04 ms`.
3. **Retrieval + generation.** The vanilla pipeline in `hybrid` mode retrieved eight chunks
   (`chunk_ids`) using the local `llama3.1:8b` model, spending `3359` input / `40` output
   tokens (`cost_usd = 0.0`, local). `retrieve` took 0.34 s, `synthesize` 1.58 s.
4. **Retrieved-content scan.** `retrieved_flagged_chunk_ids` is empty — none of the eight
   chunks tripped the injection heuristics.
5. **Output guardrail.** `output_scan.blocked = false`, no detections — the answer carried
   no PII, so it was returned unmodified. Cost: `guardrails_out = 0.04 ms`.
6. **Egress.** `refusal = false`, `refusal_reason = null`: a grounded answer was returned.
   Its text hashes to `aa45c811…`, which you can check against any stored copy without the
   log ever having retained the answer itself.

Total wall-clock `1.914 s`, of which the guardrails added under a tenth of a millisecond —
the whole governance layer is regex on the default path. Reconstructing this lifecycle used
nothing but the one log line.
