from __future__ import annotations

import pytest

from agentic_rag.config import Settings


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Isolated settings: code defaults only, data under a temp dir."""
    return Settings(data_dir=tmp_path / "data")
