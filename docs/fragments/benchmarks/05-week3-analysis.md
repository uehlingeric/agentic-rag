### Analysis: both local rerankers hurt — a negative result worth keeping

Neither local reranker improves on the hybrid baseline; both degrade every mode on nearly
every metric, and hybrid — the strongest first stage — is hit hardest (NDCG@10 0.7126 →
0.5922 with the 8B LLM, → 0.4608 with the cross-encoder). The damage is broad, not
concentrated: per-query, hybrid+llm is worse on 16 of 25 questions and better on only 4.

Two failure mechanisms, verified by direct probing rather than assumed:

1. **8B listwise judgment is weak.** The LLM reranker returned parseable rankings (a parse
   fallback would reproduce the baseline row, which is not what we see) — its orderings are
   simply worse than RRF's. Listwise reranking is known to demand strong instruction-following
   models; llama3.1:8b is below that bar.
2. **The cross-encoder confidently prefers the wrong chunks.** Probing v1-q01 (first
   requirement of AC-2): it scores PS-5 and CM-5 chunks 0.97/0.91 and the AC-2 body chunks
   0.63–0.79. The integration is correct (pair order, sigmoid scores, descending sort);
   the preference is the model's. Contributing factors: body chunks open with running-header
   boilerplate ("This publication is available free of charge from: https://doi.org/…") that
   consumes the model's 512-token window, while appendix/crosswalk chunks are keyword-dense;
   and a first stage at recall@5 0.88 leaves a reranker almost no headroom — the classic
   regime where reranking adds variance, not precision.

Cost/latency ($0, local): llm reranking averaged 3.97s/query (128.7k input / 18.8k output
tokens over 75 calls); the cross-encoder averaged 0.59s/query on GPU after model load.

Decisions: `rerank.mode` default stays `none` — the week-5 vanilla-RAG baseline will not
carry a stage that measurably hurts. Boilerplate stripping in chunk text is a week-4
candidate fix (it should help the cross-encoder and shrink synthesis prompts). Rerun this
table with an API-grade model as the listwise judge once provider keys are configured.
