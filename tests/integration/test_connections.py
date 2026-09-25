"""Connection tests: can we actually open Chroma and SQLite, and construct
the Gemini client, given the configured settings?"""

import sqlite3

import chromadb
import pytest

from llm.gemini_client import GeminiClient

EXPECTED_TABLES = {"regions", "customers", "subscriptions", "invoices"}


def test_sqlite_connects_and_has_expected_tables(seeded_db):
    conn = sqlite3.connect(seeded_db.sqlite_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in rows}
    finally:
        conn.close()

    assert EXPECTED_TABLES.issubset(table_names)


def test_chroma_client_opens_and_collection_exists(ingested_settings):
    client = chromadb.PersistentClient(path=str(ingested_settings.chroma_path))
    collection = client.get_or_create_collection(ingested_settings.chroma_collection)

    assert collection.count() > 0


def test_gemini_client_constructs_from_settings(test_settings):
    client = GeminiClient(test_settings)

    assert client.settings.gemini_model == test_settings.gemini_model


def test_gemini_client_live_call_only_runs_when_enabled(test_settings):
    if not test_settings.run_live_llm_tests:
        pytest.skip("Set RUN_LIVE_LLM_TESTS=1 and a real GEMINI_API_KEY to run this live test.")

    from config import get_settings

    live_client = GeminiClient(get_settings())
    response = live_client.generate("Reply with exactly the word: pong")
    assert "pong" in response.lower()
