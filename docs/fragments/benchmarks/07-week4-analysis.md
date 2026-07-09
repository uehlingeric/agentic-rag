### Analysis

**Read the judge column before the score columns.** Per ADR-006 an answer is never scored
by the provider that generated it, so the anthropic rows are scored by gemini-3.5-flash and
the ollama/google rows by claude-sonnet-4-6. Both judges pass calibration on all three
rubrics with `judge.v2` (κ 0.68–0.93 vs independent labels; see `docs/judge-calibration.md`
— note the labels are model-authored, not human), but they are not equally severe: on an
identical 41-answer overlap slice gemini scored +0.4 to +0.9 higher than sonnet across
dimensions. The anthropic row's flat 5.00s are partly that leniency. Under a single judge
(sonnet, self-judging included), google's hybrid answers scored slightly *higher* on
faithfulness than anthropic's (4.69 vs 4.48) while anthropic led on relevance (4.90 vs
4.62) — the two cloud providers are close to indistinguishable on answer quality here, and
this table should not be read as a provider ranking without that caveat. Judge variance is
a non-issue: two same-judge runs over one config moved aggregates by ≤ 0.024 against the
±0.2 gate.

**Refusal behavior, not answer quality, is the biggest provider difference.** The two cloud
models refuse aggressively whenever the retrieved context does not state the answer:
45% of answerable questions on bm25, falling to 26–31% on hybrid — the false-refusal rate
tracks retrieval quality almost monotonically, which is what a strictly grounded model
should do when handed a context that genuinely lacks the answer (the corpus contains it,
but retrieval did not surface it). llama3.1:8b almost never refuses (0–10%) and pays for it
on the rubrics: its faithfulness sits near 3 because it answers from weak context and the
judge finds unsupported claims. The same trade shows up on the unanswerable slice, where
the cloud models refuse 88–100% correctly versus ollama's 63–88%.

**The "bad" reranker helps end-to-end.** Week 3 showed the llama-based listwise reranker
*degrades* section-level retrieval metrics, and it was left off by default. End-to-end the
picture inverts for the cloud providers: hybrid+rerank roughly halves false refusals
(anthropic 0.26 → 0.17, google 0.31 → 0.19) at equal-or-better rubric scores — reordering
the 30-candidate pool pulls answer-bearing chunks into the top-8 context often enough to
matter, even though the ordering it produces looks worse to NDCG@10. The win is not free:
the local rerank stage adds ~6–10 s latency per question, ollama's own scores drop
(faithfulness 2.98 → 2.80), and one previously-refused unanswerable slipped through for
anthropic (refusal correctness 1.00 → 0.875). Note the configs are not context-size-matched
(rerank cuts to top-8, no-rerank to top-10), so part of the effect may be context length
rather than ordering.

**Ops.** gemini-3.5-flash is by far the fastest (≈1.5 s mean end-to-end vs ≈9–10 s for
Bedrock sonnet-4-6 and 2.5–4.4 s for local llama3.1:8b). At current list prices a full
50-question config costs ≈ $0.78–0.84 to generate with sonnet-4-6 (Bedrock `global.`
profile) and ≈ $0.35 with gemini-3.5-flash; cross-provider judging adds $0.09–0.45 per
config depending on the judge and how many answers were refusals (refusals are not judged).
Provenance for every row: dataset v2, synthesis.v2, judge.v2, per-row judge identity in the
results JSONL (`evals/results/generation-20260709-200512Z/`).
