## Reproduce

```bash
agentic-rag ingest && agentic-rag index && agentic-rag eval retrieval
agentic-rag eval rerank                             # llm reranker (ollama)
uv sync --extra dev --extra rerank-local
agentic-rag eval rerank --reranker cross-encoder    # local bge-reranker-base
```

All Week 2 and Week 3 numbers are $0 runs: BM25 is local SQLite, embeddings and the LLM
reranker are local Ollama, and the cross-encoder runs locally.
