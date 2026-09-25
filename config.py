from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time, before any Settings are constructed.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    chroma_path: Path
    chroma_collection: str
    embedding_model: str
    retrieval_similarity_threshold: float
    retrieval_top_k: int
    sqlite_path: Path
    log_path: Path
    chart_output_dir: Path
    run_live_llm_tests: bool

    @classmethod
    def from_env(cls) -> "Settings":
        def _path(env_var: str, default: str) -> Path:
            raw = os.environ.get(env_var, default)
            p = Path(raw)
            return p if p.is_absolute() else BASE_DIR / p

        return cls(
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
            chroma_path=_path("CHROMA_PATH", "data/chroma"),
            chroma_collection=os.environ.get("CHROMA_COLLECTION", "enterprise_docs"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            retrieval_similarity_threshold=float(
                os.environ.get("RETRIEVAL_SIMILARITY_THRESHOLD", "0.35")
            ),
            retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "4")),
            sqlite_path=_path("SQLITE_PATH", "data/app.db"),
            log_path=_path("LOG_PATH", "output/logs/app.log"),
            chart_output_dir=_path("CHART_OUTPUT_DIR", "output/charts"),
            run_live_llm_tests=os.environ.get("RUN_LIVE_LLM_TESTS", "0") == "1",
        )


def get_settings() -> Settings:
    """Fresh Settings on every call so tests can monkeypatch env vars per-test."""
    return Settings.from_env()
