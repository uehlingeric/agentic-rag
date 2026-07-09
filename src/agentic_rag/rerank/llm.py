"""LLM-powered listwise reranker for relevance ordering."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace

from agentic_rag.prompts import load_prompt
from agentic_rag.providers.base import LLMProvider, Message, Role, Usage
from agentic_rag.retrieval.base import ScoredChunk


class LLMReranker:
    """Reranks candidates using an LLM to judge relevance."""

    name = "llm"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        model: str | None = None,
        prompt_version: int | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._prompt = load_prompt("rerank", version=prompt_version)
        self.last_usage = Usage.zero()

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        """Rerank candidates by LLM judgment of relevance to query.

        Returns candidates reordered by LLM ranking, cut to top_k, with ranks
        1..n. Scores and source_scores are preserved. On any parse failure,
        falls back to input order.
        """
        if not candidates:
            self.last_usage = Usage.zero()
            return []

        # Format candidates for the prompt: "chunk_id: excerpt"
        lines = []
        for chunk in candidates:
            excerpt = chunk.chunk.text
            # Flatten whitespace: collapse runs to single spaces
            excerpt = re.sub(r"\s+", " ", excerpt)
            # Truncate to 200 characters
            excerpt = excerpt[:200]
            lines.append(f"{chunk.chunk.chunk_id}: {excerpt}")

        candidates_text = "\n".join(lines)

        # Render prompt and call LLM
        prompt_text = self._prompt.render(query=query, candidates=candidates_text)
        message = Message(role=Role.USER, content=prompt_text)
        completion = await self._llm.complete(
            [message],
            model=self._model,
            temperature=0.0,
            max_tokens=1024,
        )
        self.last_usage = completion.usage

        # Parse response
        ranking = self._parse_ranking(completion.text)

        if ranking is None:
            # Fallback: input order, cut to top_k, reassign ranks
            return [replace(c, rank=i) for i, c in enumerate(candidates[:top_k], start=1)]

        # Reorder: ranked ids first, then remaining candidates in input order
        unranked_chunks = {c.chunk.chunk_id: c for c in candidates}

        result = []
        seen_ids = set()

        # Add ranked ids in order (skip unknown and duplicates)
        for chunk_id in ranking:
            if chunk_id in unranked_chunks and chunk_id not in seen_ids:
                result.append(unranked_chunks[chunk_id])
                seen_ids.add(chunk_id)

        # Append remaining candidates in input order
        for chunk in candidates:
            if chunk.chunk.chunk_id not in seen_ids:
                result.append(chunk)
                seen_ids.add(chunk.chunk.chunk_id)

        # Cut to top_k and reassign ranks
        return [replace(c, rank=i) for i, c in enumerate(result[:top_k], start=1)]

    def _parse_ranking(self, text: str) -> list[str] | None:
        """Parse LLM response to ranking list, or return None on failure.

        Strips markdown code fences if present. Returns None if JSON is
        invalid, missing "ranking" key, ranking is not a list, or entries
        are not strings.
        """
        # Strip markdown code fences
        stripped = text.strip()
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or "ranking" not in data:
            return None

        ranking = data["ranking"]
        if not isinstance(ranking, list):
            return None

        if not all(isinstance(item, str) for item in ranking):
            return None

        return ranking
