"""Download and verify NIST corpus documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from agentic_rag.config import Settings
from agentic_rag.ingest.sources import SOURCES


class ChecksumMismatchError(Exception):
    """Raised when downloaded file checksum does not match pinned hash."""

    pass


@dataclass
class DownloadedDoc:
    """Metadata for a successfully downloaded document."""

    doc_id: str
    path: Path
    sha256: str
    size_bytes: int


def download_corpus(
    settings: Settings, doc_ids: list[str] | None = None, force: bool = False
) -> list[DownloadedDoc]:
    """Download and verify NIST corpus documents.

    Args:
        settings: Application settings with data_dir.
        doc_ids: Restrict to specific doc_ids (default: all in SOURCES).
        force: Re-download even if file exists with matching sha256.

    Returns:
        List of DownloadedDoc with verified paths and hashes.

    Raises:
        ChecksumMismatchError: If downloaded file hash does not match pinned sha256.
        httpx.HTTPError: On network errors (non-200 status, timeout, etc).
    """
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    doc_ids_to_download = doc_ids or list(SOURCES.keys())
    results = []

    client = httpx.Client(follow_redirects=True, timeout=60.0)

    for doc_id in doc_ids_to_download:
        source = SOURCES[doc_id]
        output_path = raw_dir / f"{doc_id}.pdf"

        # Check if file exists and matches pinned hash
        if output_path.exists() and not force:
            existing_sha256 = _compute_file_sha256(output_path)
            if existing_sha256 == source.sha256:
                results.append(
                    DownloadedDoc(
                        doc_id=doc_id,
                        path=output_path,
                        sha256=existing_sha256,
                        size_bytes=output_path.stat().st_size,
                    )
                )
                continue

        # Download
        response = client.get(source.url)
        response.raise_for_status()  # Raises on non-200

        data = response.content
        downloaded_sha256 = hashlib.sha256(data).hexdigest()

        # Verify checksum
        if downloaded_sha256 != source.sha256:
            output_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"{doc_id}: computed {downloaded_sha256}, expected {source.sha256}"
            )

        # Write to disk
        output_path.write_bytes(data)

        results.append(
            DownloadedDoc(
                doc_id=doc_id,
                path=output_path,
                sha256=downloaded_sha256,
                size_bytes=len(data),
            )
        )

    return results


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
