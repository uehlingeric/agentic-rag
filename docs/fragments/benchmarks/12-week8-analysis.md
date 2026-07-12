### Analysis

- **The agent loop's refusal win generalizes across retrieval configs.** Week 5
  measured it only at hybrid+llm; the full matrix shows the loop cutting false
  refusals for both cloud providers in **all four** retrieval configs — anthropic
  mean 0.28 → 0.18, google 0.37 → 0.26 — with the largest gains where retrieval is
  weakest (bm25: anthropic 0.45 → 0.24, google 0.47 → 0.26). Decomposition partially
  compensates for a weak retriever. Rubric means move ≤ 0.06 for cloud providers,
  at roughly 2× generation cost and ~2× p50 latency.
- **Guardrails-on is free at benchmark scale.** 0 of 1,104 rows were refused by a
  guardrail (input or output); every refusal in the tables is the model's own
  grounded refusal. This is the production path — the numbers above are what the
  API serves.
- **Stability vs. the week-4/5 runs:** every cloud-provider rubric delta is ≤ 0.14
  — the prompt hardening (week 6) did not move cloud quality. The refusal-rate
  differences per config are variance-level (±1–5 flips of 46).
- **One investigated anomaly (8B only):** ollama citation accuracy dropped 0.3–0.5
  under the delimiter-hardened synthesis prompts (bm25 vanilla 3.36 → 2.84). Same
  judge both runs; per-row justifications show llama3.1:8b now sometimes writes
  "excerpt id=6" in prose instead of `[6]` markers, or attaches the wrong excerpt
  id — a citation-format regression induced by the `<excerpt id=n>` context format
  that cloud models are immune to (their citation accuracy held at ≈ 5.0). The
  security hardening is kept; the cost lands entirely on the smallest model and is
  documented in [limitations.md](limitations.md).
- **The loop still hurts the 8B model** (replicating week 5, now across all
  configs): faithfulness 3.00 → 2.26 and citation accuracy 3.00 → 2.03 at
  hybrid+llm. Run the agent loop on models with headroom; run 8B vanilla.
- **Measured run cost:** $23.67 ($17.39 generation + $6.28 judging) for 1,104
  generation+judge pairs, per-row usage accounting.
