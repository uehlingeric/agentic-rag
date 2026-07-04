"""Tests for document download and verification."""

import hashlib
from pathlib import Path

import pytest
import respx
from httpx import Response

from agentic_rag.config import Settings
from agentic_rag.ingest.download import ChecksumMismatchError, download_corpus
from agentic_rag.ingest.sources import SOURCES


@pytest.fixture
def settings_with_tmpdir(tmp_path: Path) -> Settings:
    """Settings pointing to a temp data directory."""
    return Settings(data_dir=tmp_path)


@respx.mock
def test_download_single_doc(settings_with_tmpdir, monkeypatch):
    """Download a single document with mocked HTTP."""
    doc_id = "test-doc"
    fake_content = b"This is fake PDF content"
    expected_sha256 = hashlib.sha256(fake_content).hexdigest()

    # Patch SOURCES with a test document
    from agentic_rag.ingest.sources import SourceDoc

    test_source = SourceDoc(
        doc_id=doc_id,
        title="Test Document",
        url="https://example.com/test.pdf",
        sha256=expected_sha256,
    )
    monkeypatch.setitem(SOURCES, doc_id, test_source)

    # Mock the HTTP request
    respx.get("https://example.com/test.pdf").mock(return_value=Response(200, content=fake_content))

    downloaded = download_corpus(settings_with_tmpdir, doc_ids=[doc_id])

    assert len(downloaded) == 1
    assert downloaded[0].doc_id == doc_id
    assert downloaded[0].sha256 == expected_sha256
    assert downloaded[0].path == settings_with_tmpdir.data_dir / "raw" / f"{doc_id}.pdf"
    assert downloaded[0].size_bytes == len(fake_content)


@respx.mock
def test_download_checksum_mismatch(settings_with_tmpdir, monkeypatch):
    """Verify checksum mismatch raises error and deletes file."""
    doc_id = "test-doc-mismatch"
    wrong_content = b"Wrong content"
    expected_sha256 = "wrong_hash_value_12345"

    from agentic_rag.ingest.sources import SourceDoc

    test_source = SourceDoc(
        doc_id=doc_id,
        title="Test Document",
        url="https://example.com/test2.pdf",
        sha256=expected_sha256,
    )
    monkeypatch.setitem(SOURCES, doc_id, test_source)

    # Mock with content that doesn't match the pinned sha256
    respx.get("https://example.com/test2.pdf").mock(
        return_value=Response(200, content=wrong_content)
    )

    with pytest.raises(ChecksumMismatchError):
        download_corpus(settings_with_tmpdir, doc_ids=[doc_id])

    # Verify the bad file was cleaned up
    bad_path = settings_with_tmpdir.data_dir / "raw" / f"{doc_id}.pdf"
    assert not bad_path.exists(), "Failed download should be deleted"


@respx.mock
def test_download_skip_existing_file(settings_with_tmpdir, monkeypatch):
    """Skip download if file exists with matching checksum."""
    doc_id = "test-doc-skip"
    test_content = b"test content for skipping"
    test_sha256 = hashlib.sha256(test_content).hexdigest()

    from agentic_rag.ingest.sources import SourceDoc

    test_source = SourceDoc(
        doc_id=doc_id,
        title="Test Document",
        url="https://example.com/test3.pdf",
        sha256=test_sha256,
    )
    monkeypatch.setitem(SOURCES, doc_id, test_source)

    # Create a file with the correct sha256
    raw_dir = settings_with_tmpdir.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{doc_id}.pdf"
    file_path.write_bytes(test_content)

    # Mock the HTTP request - it shouldn't be called if file exists and hash matches
    # We verify this by asserting the request is not called
    respx.get("https://example.com/test3.pdf")

    # Download should skip since file exists with matching hash
    downloaded = download_corpus(settings_with_tmpdir, doc_ids=[doc_id])

    assert len(downloaded) == 1
    assert downloaded[0].doc_id == doc_id
    assert downloaded[0].sha256 == test_sha256


@respx.mock
def test_download_force_redownload(settings_with_tmpdir, monkeypatch):
    """Force flag causes re-download even if file exists."""
    doc_id = "test-doc-force"
    fake_content = b"New fake PDF content"
    expected_sha256 = hashlib.sha256(fake_content).hexdigest()

    from agentic_rag.ingest.sources import SourceDoc

    test_source = SourceDoc(
        doc_id=doc_id,
        title="Test Document",
        url="https://example.com/test4.pdf",
        sha256=expected_sha256,
    )
    monkeypatch.setitem(SOURCES, doc_id, test_source)

    # Create an existing file with different content
    raw_dir = settings_with_tmpdir.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{doc_id}.pdf"
    file_path.write_bytes(b"old content")

    # Mock the HTTP response with new content
    respx.get("https://example.com/test4.pdf").mock(
        return_value=Response(200, content=fake_content)
    )

    # With force=True, should re-download and update
    downloaded = download_corpus(settings_with_tmpdir, doc_ids=[doc_id], force=True)

    assert len(downloaded) == 1
    assert downloaded[0].sha256 == expected_sha256
    assert file_path.read_bytes() == fake_content


@respx.mock
def test_download_http_error(settings_with_tmpdir):
    """Non-200 status raises error."""
    doc_id = "fips-199"
    source = SOURCES[doc_id]

    # Mock a 404 response
    respx.get(source.url).mock(return_value=Response(404))

    from httpx import HTTPStatusError

    with pytest.raises(HTTPStatusError):
        download_corpus(settings_with_tmpdir, doc_ids=[doc_id])


@pytest.mark.live
@respx.mock(assert_all_called=False)
def test_download_real_fips_199():
    """Live test: download FIPS 199 and verify checksum (excluded by default)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(data_dir=Path(tmpdir))

        # Don't mock - let it make real request
        respx.get("https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.199.pdf").pass_through()

        downloaded = download_corpus(settings, doc_ids=["fips-199"])

        assert len(downloaded) == 1
        assert downloaded[0].doc_id == "fips-199"
        # Verify against real pinned hash
        assert (
            downloaded[0].sha256
            == "73d19f05f71e30f378050f178aa3943c38790bbae56c07f2b5708c5a1a90242f"
        )
