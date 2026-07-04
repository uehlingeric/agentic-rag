"""Tests for the ingestion pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import respx

from agentic_rag.config import Settings
from agentic_rag.ingest.extract import Section
from agentic_rag.ingest.pipeline import run_ingest


@pytest.fixture
def fixture_sections() -> list[Section]:
    """Fixture sections for testing pipeline."""
    return [
        Section(
            section_id="1",
            section_path="Chapter 1",
            heading="Introduction",
            page_start=0,
            page_end=2,
            text="This is the introduction section with content. " * 20,
        ),
        Section(
            section_id="AC-2",
            section_path="Controls > Access Control",
            heading="AC-2 Account Management",
            page_start=5,
            page_end=8,
            text="This control addresses account management. " * 30,
        ),
    ]


@respx.mock
def test_pipeline_end_to_end(tmp_path: Path, fixture_sections):
    """End-to-end pipeline test with mocked download and patched extract."""
    settings = Settings(data_dir=tmp_path)

    # Create a minimal fake PDF file
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fake_pdf = raw_dir / "test-doc.pdf"
    fake_pdf.write_bytes(b"fake pdf content")

    # Mock the download to just return our fake file
    with patch("agentic_rag.ingest.pipeline.download_corpus") as mock_download:
        from agentic_rag.ingest.download import DownloadedDoc

        mock_download.return_value = [
            DownloadedDoc(
                doc_id="test-doc",
                path=fake_pdf,
                sha256="abc123",
                size_bytes=15,
            )
        ]

        # Mock extract to return fixtures
        with patch("agentic_rag.ingest.pipeline.extract_sections") as mock_extract:
            mock_extract.return_value = fixture_sections

            # Run pipeline
            manifest = run_ingest(settings, doc_ids=["test-doc"])

            # Verify output
            assert manifest.total_chunks > 0
            assert len(manifest.documents) == 1
            assert manifest.documents[0].doc_id == "test-doc"
            assert manifest.documents[0].sections == len(fixture_sections)

            # Verify files were written
            chunks_path = tmp_path / "corpus" / "chunks.jsonl"
            manifest_path = tmp_path / "corpus" / "manifest.json"

            assert chunks_path.exists(), "chunks.jsonl should be written"
            assert manifest_path.exists(), "manifest.json should be written"

            # Verify chunks.jsonl format
            with open(chunks_path) as f:
                for line in f:
                    chunk_dict = json.loads(line)
                    # Verify all required fields are present
                    required_fields = [
                        "chunk_id",
                        "doc_id",
                        "section_id",
                        "text",
                        "token_count",
                    ]
                    for field in required_fields:
                        assert field in chunk_dict, f"Missing field: {field}"

            # Verify manifest.json
            with open(manifest_path) as f:
                manifest_data = json.load(f)
                assert "total_chunks" in manifest_data
                assert "documents" in manifest_data
                assert "created_at" in manifest_data
                assert "corpus_version" in manifest_data
                assert manifest_data["corpus_version"] == "v1"


@respx.mock
def test_pipeline_subset_docs(tmp_path: Path, fixture_sections):
    """Pipeline with doc_ids subset only processes those docs."""
    settings = Settings(data_dir=tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with patch("agentic_rag.ingest.pipeline.download_corpus") as mock_download:
        from agentic_rag.ingest.download import DownloadedDoc

        # Return only the requested subset
        mock_download.return_value = [
            DownloadedDoc(
                doc_id="doc1",
                path=raw_dir / "doc1.pdf",
                sha256="abc",
                size_bytes=100,
            )
        ]

        (raw_dir / "doc1.pdf").write_bytes(b"fake")

        with patch("agentic_rag.ingest.pipeline.extract_sections") as mock_extract:
            mock_extract.return_value = fixture_sections

            manifest = run_ingest(settings, doc_ids=["doc1"])

            # Verify only one doc was processed
            assert len(manifest.documents) == 1
            assert manifest.documents[0].doc_id == "doc1"

            # Verify download was called with correct subset
            mock_download.assert_called_once()
            assert mock_download.call_args[1]["doc_ids"] == ["doc1"]


def test_pipeline_creates_output_directories(tmp_path: Path):
    """Pipeline creates necessary output directories."""
    settings = Settings(data_dir=tmp_path)

    with patch("agentic_rag.ingest.pipeline.download_corpus") as mock_download:
        # Return empty list
        mock_download.return_value = []

        run_ingest(settings, doc_ids=[])

        # Even with no docs, directories should exist
        corpus_dir = tmp_path / "corpus"
        assert corpus_dir.exists(), "corpus directory should be created"
        assert (corpus_dir / "chunks.jsonl").exists(), "chunks.jsonl should exist"
        assert (corpus_dir / "manifest.json").exists(), "manifest.json should exist"


def test_manifest_structure(tmp_path: Path, fixture_sections):
    """Verify manifest structure matches spec."""
    settings = Settings(data_dir=tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "test.pdf").write_bytes(b"fake")

    with patch("agentic_rag.ingest.pipeline.download_corpus") as mock_download:
        from agentic_rag.ingest.download import DownloadedDoc

        mock_download.return_value = [
            DownloadedDoc(
                doc_id="test",
                path=raw_dir / "test.pdf",
                sha256="abc123",
                size_bytes=100,
            )
        ]

        with patch("agentic_rag.ingest.pipeline.extract_sections") as mock_extract:
            mock_extract.return_value = fixture_sections

            manifest = run_ingest(settings)

            # Check required manifest fields
            assert manifest.documents is not None
            assert manifest.total_chunks >= 0
            assert manifest.output_path is not None
            assert manifest.target_tokens == 512
            assert manifest.overlap_tokens == 64
            assert manifest.tokenizer == "o200k_base"
            assert manifest.created_at is not None
            assert manifest.corpus_version == "v1"

            # Verify ISO8601 timestamp
            assert "T" in manifest.created_at  # ISO format has T
            assert "Z" in manifest.created_at or "+" in manifest.created_at  # Timezone aware


def test_chunks_jsonl_schema(tmp_path: Path, fixture_sections):
    """Verify chunks.jsonl has correct schema."""
    settings = Settings(data_dir=tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "test.pdf").write_bytes(b"fake")

    with patch("agentic_rag.ingest.pipeline.download_corpus") as mock_download:
        from agentic_rag.ingest.download import DownloadedDoc

        mock_download.return_value = [
            DownloadedDoc(
                doc_id="test",
                path=raw_dir / "test.pdf",
                sha256="abc123",
                size_bytes=100,
            )
        ]

        with patch("agentic_rag.ingest.pipeline.extract_sections") as mock_extract:
            mock_extract.return_value = fixture_sections

            run_ingest(settings)

            # Read and verify chunks.jsonl
            chunks_path = tmp_path / "corpus" / "chunks.jsonl"
            chunks = []
            with open(chunks_path) as f:
                for line in f:
                    chunks.append(json.loads(line))

            assert len(chunks) > 0

            # Verify schema of first chunk
            chunk = chunks[0]
            required_fields = {
                "chunk_id": str,
                "doc_id": str,
                "section_id": str,
                "section_path": str,
                "heading": str,
                "page_start": int,
                "page_end": int,
                "char_start": int,
                "char_end": int,
                "token_count": int,
                "content_type": str,
                "text": str,
            }

            for field, field_type in required_fields.items():
                assert field in chunk, f"Missing field: {field}"
                assert isinstance(chunk[field], field_type), f"Field {field} has wrong type"
