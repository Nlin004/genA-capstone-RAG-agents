"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")


class CitationSchema(BaseModel):
    doc_id: str
    similarity: float
    snippet: str


class AnswerPartSchema(BaseModel):
    agent: str
    answer: str


class QueryResponse(BaseModel):
    final_answer: str
    parts: list[AnswerPartSchema] = []
    clarification_needed: bool = False
    clarification_question: str | None = None


class QualitativeResponse(BaseModel):
    answer: str
    citations: list[CitationSchema] = []
    used_fallback: bool = False


class QuantitativeResponse(BaseModel):
    answer: str
    sql: str | None = None
    rows: list[dict] = []
    chart_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    chroma_ok: bool
    sqlite_ok: bool
