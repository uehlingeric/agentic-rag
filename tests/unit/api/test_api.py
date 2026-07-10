"""Tests for the FastAPI service."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from httpx import AsyncClient

from agentic_rag.api import create_app
from agentic_rag.config import Settings


class TestHealth:
    """Tests for GET /health (no auth)."""

    async def test_health_ok(self, client: AsyncClient) -> None:
        """Health endpoint returns 200 with status ok."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True


class TestAuthentication:
    """Tests for authorization header validation."""

    async def test_ask_missing_auth(self, client: AsyncClient) -> None:
        """Missing Authorization header returns 401."""
        response = await client.post("/ask", json={"question": "Test question"})
        assert response.status_code == 401
        assert "application/problem+json" in response.headers.get("content-type", "")
        assert response.headers.get("www-authenticate") == "Bearer"

    async def test_ask_invalid_bearer_format(self, client: AsyncClient) -> None:
        """Invalid Authorization format returns 401."""
        response = await client.post(
            "/ask",
            json={"question": "Test question"},
            headers={"Authorization": "InvalidFormat"},
        )
        assert response.status_code == 401

    async def test_ask_wrong_token(self, client: AsyncClient) -> None:
        """Wrong token returns 401."""
        response = await client.post(
            "/ask",
            json={"question": "Test question"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401


class TestAskVanilla:
    """Tests for POST /ask vanilla pipeline (non-streaming)."""

    async def test_ask_vanilla_happy_path(self, client: AsyncClient) -> None:
        """POST /ask vanilla returns 200 with record shape."""
        response = await client.post(
            "/ask",
            json={"question": "What is testing?"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify record shape
        assert data["question"] == "What is testing?"
        assert data["provider"] == "stub"
        assert data["mode"] == "hybrid"
        assert data["rerank"] == "none"
        assert data["pipeline"] == "vanilla"
        assert isinstance(data["answer"], str)
        assert "[1]" in data["answer"]  # Citation marker resolved
        assert data["refusal"] is False
        assert data["refusal_reason"] is None
        assert isinstance(data["citations"], list)
        assert isinstance(data["usage"], dict)
        assert data["usage"]["input_tokens"] > 0
        assert data["request_id"]
        assert data["guardrails"]

    async def test_ask_with_pii_refusal(self, client: AsyncClient) -> None:
        """POST /ask with PII question returns 200 with refusal."""
        response = await client.post(
            "/ask",
            json={"question": "My SSN is 123-45-6789 what is this?"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["refusal"] is True
        assert data["refusal_reason"] in ("input_pii", "input_injection")
        assert data["citations"] == []

    async def test_ask_agentic_pipeline(self, client: AsyncClient) -> None:
        """POST /ask with pipeline=agentic returns agent metadata."""
        response = await client.post(
            "/ask",
            json={"question": "Test question", "pipeline": "agentic"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["pipeline"] == "agentic"
        assert data["agent"] is not None
        assert "plan" in data["agent"]
        assert "sub_queries" in data["agent"]
        assert "revisions" in data["agent"]
        assert "caveat" in data["agent"]

    async def test_ask_agentic_stream_invalid(self, client: AsyncClient) -> None:
        """POST /ask agentic+stream returns 422."""
        response = await client.post(
            "/ask",
            json={
                "question": "Test question",
                "pipeline": "agentic",
                "stream": True,
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    async def test_ask_vanilla_stream(self, client: AsyncClient) -> None:
        """POST /ask vanilla with stream=true returns SSE."""
        response = await client.post(
            "/ask",
            json={"question": "Test question", "stream": True},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE frames
        content = response.text
        assert "event: delta" in content
        assert "event: result" in content

        # Extract result event
        result_match = re.search(r"event: result\ndata: ({.*?})\n\n", content, re.DOTALL)
        assert result_match is not None
        result_json = json.loads(result_match.group(1))
        assert result_json["request_id"]
        assert result_json["guardrails"]

    async def test_ask_missing_question(self, client: AsyncClient) -> None:
        """POST /ask without question returns 422."""
        response = await client.post(
            "/ask",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    async def test_ask_invalid_mode(self, client: AsyncClient) -> None:
        """POST /ask with invalid mode returns 422."""
        response = await client.post(
            "/ask",
            json={"question": "Test", "mode": "invalid"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    async def test_ask_unknown_provider(self, client: AsyncClient) -> None:
        """POST /ask with unknown provider returns 422."""
        response = await client.post(
            "/ask",
            json={"question": "Test", "provider": "nonexistent"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422


class TestSearch:
    """Tests for GET /search."""

    async def test_search_happy_path(self, client: AsyncClient) -> None:
        """GET /search returns ranked results."""
        response = await client.get(
            "/search?q=testing",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        results = response.json()

        assert isinstance(results, list)
        assert len(results) > 0

        # Check result structure
        r = results[0]
        assert "rank" in r
        assert "score" in r
        assert "chunk_id" in r
        assert "doc_id" in r
        assert "text" in r

    async def test_search_missing_query(self, client: AsyncClient) -> None:
        """GET /search without q param returns 422."""
        response = await client.get(
            "/search",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    async def test_search_top_k_validation(self, client: AsyncClient) -> None:
        """GET /search with top_k=999 returns 422."""
        response = await client.get(
            "/search?q=test&top_k=999",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    async def test_search_invalid_mode(self, client: AsyncClient) -> None:
        """GET /search with invalid mode returns 422."""
        response = await client.get(
            "/search?q=test&mode=invalid",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422


class TestStats:
    """Tests for GET /stats."""

    async def test_stats_after_request(self, client: AsyncClient) -> None:
        """GET /stats after /ask returns aggregated metrics."""
        # Make a request first
        await client.post(
            "/ask",
            json={"question": "Test question"},
            headers={"Authorization": "Bearer test-token"},
        )

        # Query stats
        response = await client.get(
            "/stats?by=provider",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["summary"], list)
        # Should have at least one row for stub provider
        summary_row = data["summary"][0]
        assert summary_row["group"] == "stub"
        assert summary_row["requests"] >= 1

    async def test_stats_with_stages(self, client: AsyncClient) -> None:
        """GET /stats?stages=true includes per-stage breakdowns."""
        await client.post(
            "/ask",
            json={"question": "Test"},
            headers={"Authorization": "Bearer test-token"},
        )

        response = await client.get(
            "/stats?by=provider&stages=true",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["stages"] is not None
        assert isinstance(data["stages"], list)

    async def test_stats_by_source(self, client: AsyncClient) -> None:
        """GET /stats?by=source shows api vs cli."""
        await client.post(
            "/ask",
            json={"question": "Test"},
            headers={"Authorization": "Bearer test-token"},
        )

        response = await client.get(
            "/stats?by=source",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()

        # Should have "api" source group
        groups = [row["group"] for row in data["summary"]]
        assert "api" in groups

    async def test_stats_no_data_yet(self, client: AsyncClient) -> None:
        """GET /stats before any request returns 404 problem+json."""
        response = await client.get(
            "/stats",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 404
        assert "application/problem+json" in response.headers.get("content-type", "")


class TestRateLimit:
    """Tests for rate limiting."""

    async def test_rate_limit_exceeded(self, settings: Settings, stub_retriever: Any) -> None:
        """Rate limit threshold triggers 429."""
        settings.api.rate_limit = "2/minute"
        app = create_app(settings, retriever=stub_retriever)

        from httpx import ASGITransport

        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

        # Make two requests (should succeed)
        for _ in range(2):
            response = await client.post(
                "/ask",
                json={"question": "Test"},
                headers={"Authorization": "Bearer test-token"},
            )
            assert response.status_code == 200

        # Third request (should be rate limited)
        response = await client.post(
            "/ask",
            json={"question": "Test"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 429


class TestStartupGuard:
    """Tests for startup validation."""

    def test_create_app_no_token(self, settings: Settings) -> None:
        """create_app raises RuntimeError when api.token is None."""
        settings.api.token = None
        with pytest.raises(RuntimeError, match="AGENTIC_RAG_API__TOKEN"):
            create_app(settings)
