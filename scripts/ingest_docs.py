"""Chunks the markdown docs in data/docs/, embeds them with
sentence-transformers, and upserts them into a persistent Chroma
collection.

Run manually once (or whenever the docs change):

    python scripts/ingest_docs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker over paragraphs.

    Splits on blank lines first (so we don't cut mid-sentence when a
    paragraph fits), then falls back to a fixed-size sliding window for any
    paragraph longer than chunk_size.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 1 <= chunk_size:
            buffer = f"{buffer}\n{paragraph}".strip()
            continue

        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(paragraph) <= chunk_size:
            buffer = paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end])
                start = end - overlap

    if buffer:
        chunks.append(buffer)

    return chunks


def ingest(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(settings.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    # Explicit cosine space so similarity scores (1 - distance) are
    # well-defined and comparable to the RETRIEVAL_SIMILARITY_THRESHOLD.
    collection = client.get_or_create_collection(
        settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"No markdown docs found in {DOCS_DIR}")

    ids, documents, metadatas = [], [], []

    for doc_path in doc_paths:
        doc_id = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(chunk_text(text)):
            ids.append(f"{doc_id}::{chunk_index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "source_file": doc_path.name,
                    "chunk_index": chunk_index,
                }
            )

    embeddings = embedder.encode(documents, show_progress_bar=False).tolist()

    # Upsert so re-running ingestion after editing a doc doesn't duplicate rows.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return len(ids)


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks from {DOCS_DIR}")
