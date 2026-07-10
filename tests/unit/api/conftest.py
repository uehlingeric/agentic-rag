"""Fixtures for API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.api import create_app
from agentic_rag.config import ApiSettings, Settings
from agentic_rag.retrieval.base import ChunkRecord, RetrievalMode, ScoredChunk


class StubRetriever:
    """Stub retriever returning canned chunks."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks

    async def retrieve(
        self, query: str, *, mode: RetrievalMode = RetrievalMode.HYBRID, top_k: int = 10
    ) -> list[ScoredChunk]:
        """Return first top_k chunks as scored results."""
        return [
            ScoredChunk(chunk=chunk, score=0.9 - (i * 0.01), rank=i + 1)
            for i, chunk in enumerate(self.chunks[:top_k])
        ]


def make_chunk(chunk_id: str, text: str = "Test content.") -> ChunkRecord:
    """Create a test chunk."""
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="test-doc",
        section_id="test-section",
        section_ids=["test-section"],
        section_path="Test Section",
        heading="Test Heading",
        page_start=1,
        page_end=1,
        token_count=10,
        text=text,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create test settings with stub provider and local paths."""
    api_settings = ApiSettings(token="test-token", rate_limit="1000/minute")
    return Settings(
        provider="stub",
        embedding__provider="stub",
        data_dir=tmp_path / "data",
        guardrails__enabled=True,
        guardrails__audit_enabled=True,
        guardrails__audit_dir=tmp_path / "audit",
        metrics__enabled=True,
        metrics__db_path=tmp_path / "metrics.db",
        api=api_settings,
    )


@pytest.fixture
def stub_retriever() -> StubRetriever:
    """Create a stub retriever with test chunks."""
    chunks = [
        make_chunk("chunk-1", "First chunk about testing."),
        make_chunk("chunk-2", "Second chunk about APIs."),
        make_chunk("chunk-3", "Third chunk about Python."),
    ]
    return StubRetriever(chunks)


@pytest.fixture
async def client(settings: Settings, stub_retriever: StubRetriever) -> AsyncClient:
    """Create an async HTTP client for the test app."""
    app = create_app(settings, retriever=stub_retriever)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")
