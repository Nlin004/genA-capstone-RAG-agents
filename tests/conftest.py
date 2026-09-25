"""Shared pytest fixtures: an isolated Settings pointing at tmp_path,
a seeded dummy SQLite DB, an ingested Chroma store, and a mocked
GeminiClient so unit/integration tests never make real network calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import Settings
from llm.gemini_client import GeminiClient
from scripts.ingest_docs import ingest
from scripts.setup_dummy_db import build_and_seed


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-1.5-flash",
        chroma_path=tmp_path / "chroma",
        chroma_collection="test_docs",
        embedding_model="all-MiniLM-L6-v2",
        retrieval_similarity_threshold=0.35,
        retrieval_top_k=4,
        sqlite_path=tmp_path / "app.db",
        log_path=tmp_path / "logs" / "app.log",
        chart_output_dir=tmp_path / "charts",
        run_live_llm_tests=False,
    )


@pytest.fixture
def seeded_db(test_settings: Settings) -> Settings:
    build_and_seed(test_settings.sqlite_path, seed=42, num_customers=40)
    return test_settings


@pytest.fixture(scope="session")
def _ingested_chroma_path(tmp_path_factory):
    """Ingest the real data/docs/ once per test session (embedding is slow)
    into a session-scoped temp Chroma store, reused read-only by tests."""
    chroma_dir = tmp_path_factory.mktemp("chroma_session")
    settings = Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-1.5-flash",
        chroma_path=chroma_dir,
        chroma_collection="test_docs",
        embedding_model="all-MiniLM-L6-v2",
        retrieval_similarity_threshold=0.35,
        retrieval_top_k=4,
        sqlite_path=chroma_dir / "unused.db",
        log_path=chroma_dir / "logs" / "app.log",
        chart_output_dir=chroma_dir / "charts",
        run_live_llm_tests=False,
    )
    ingest(settings)
    return chroma_dir


@pytest.fixture
def ingested_settings(test_settings: Settings, _ingested_chroma_path) -> Settings:
    """test_settings, but chroma_path repointed at the pre-ingested store."""
    return Settings(
        **{**test_settings.__dict__, "chroma_path": _ingested_chroma_path}
    )


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    client = MagicMock(spec=GeminiClient)
    client.generate.return_value = "mocked answer"
    return client
