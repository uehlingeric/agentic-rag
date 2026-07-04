You are reranking search results from NIST security publications for relevance to a query.

Query: ${query}

Candidates, one per line as `id: excerpt`:
${candidates}

Respond with ONLY a JSON object, no prose, no code fences, of exactly this form:
{"ranking": ["<id>", "<id>", ...]}

The list must contain ALL candidate ids, ordered from most to least relevant to the query. Judge relevance by whether the excerpt actually answers or supports answering the query, not by keyword overlap alone.
