"""Qualitative RAG Agent: semantic search over the enterprise docs in
Chroma, then a Gemini-generated answer grounded in the retrieved chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings
from llm.gemini_client import GeminiClient, GeminiError
from logging_config import get_logger

logger = get_logger(__name__)

NOT_FOUND_MESSAGE = (
    "I couldn't find anything in the knowledge base relevant enough to "
    "answer that confidently."
)


@dataclass
class Citation:
    doc_id: str
    similarity: float
    snippet: str


@dataclass
class QualitativeResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    used_fallback: bool = False


class QualitativeAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        gemini_client: GeminiClient | None = None,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gemini_client = gemini_client or GeminiClient(self.settings)
        self.embedder = embedder or SentenceTransformer(self.settings.embedding_model)

        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.collection = client.get_or_create_collection(
            self.settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def _retrieve(self, query: str) -> list[Citation]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.settings.retrieval_top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        citations: list[Citation] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            similarity = 1.0 - distance  # cosine space: distance = 1 - cosine similarity
            if similarity >= self.settings.retrieval_similarity_threshold:
                citations.append(
                    Citation(
                        doc_id=metadata["doc_id"],
                        similarity=round(similarity, 4),
                        snippet=document,
                    )
                )
        return citations

    def answer(self, query: str) -> QualitativeResult:
        start = time.monotonic()
        citations = self._retrieve(query)

        if not citations:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(
                "qualitative_no_relevant_chunks",
                extra={"query": query, "elapsed_ms": elapsed_ms},
            )
            return QualitativeResult(answer=NOT_FOUND_MESSAGE, citations=[], used_fallback=True)

        context = "\n\n".join(
            f"[Source: {c.doc_id}, similarity={c.similarity}]\n{c.snippet}" for c in citations
        )
        prompt = (
            "Answer the question using ONLY the information in the sources below. "
            "If the sources don't fully answer the question, say what is missing. "
            "Be concise.\n\n"
            f"SOURCES:\n{context}\n\nQUESTION: {query}"
        )

        try:
            answer_text = self.gemini_client.generate(prompt)
        except GeminiError:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.error(
                "qualitative_generation_failed",
                extra={"query": query, "elapsed_ms": elapsed_ms},
            )
            return QualitativeResult(
                answer="I found relevant sources but couldn't generate an answer right now.",
                citations=citations,
                used_fallback=True,
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "qualitative_answer_success",
            extra={
                "query": query,
                "elapsed_ms": elapsed_ms,
                "sources": [c.doc_id for c in citations],
            },
        )
        return QualitativeResult(answer=answer_text, citations=citations, used_fallback=False)
