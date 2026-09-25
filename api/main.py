"""FastAPI service exposing the Manager, Qualitative, and Quantitative
agents over HTTP.

Run with:
    uvicorn api.main:app --reload

Interactive docs at /docs, OpenAPI schema at /openapi.json, health at
/health.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from fastapi import Depends, FastAPI

from agents.manager_agent import ManagerAgent
from agents.qualitative_agent import QualitativeAgent
from agents.quantitative_agent import QuantitativeAgent
from api.schemas import (
    AnswerPartSchema,
    CitationSchema,
    HealthResponse,
    QualitativeResponse,
    QuantitativeResponse,
    QueryRequest,
    QueryResponse,
)
from config import get_settings

app = FastAPI(
    title="Multi-Agent RAG System for Enterprise Documentation",
    description="Manager, Qualitative RAG, and Quantitative NL-to-SQL agents over HTTP.",
    version="1.0.0",
)


# Dependency accessors, lazily constructed and cached on first real use.
# Routes depend on these rather than building agents at import/startup time,
# so `app.dependency_overrides` can swap in mocks in tests without ever
# constructing a real embedder/Chroma client/Gemini client.
@lru_cache
def get_qualitative_agent() -> QualitativeAgent:
    return QualitativeAgent(get_settings())


@lru_cache
def get_quantitative_agent() -> QuantitativeAgent:
    return QuantitativeAgent(get_settings())


@lru_cache
def get_manager_agent() -> ManagerAgent:
    settings = get_settings()
    return ManagerAgent(
        settings=settings,
        qualitative_agent=get_qualitative_agent(),
        quantitative_agent=get_quantitative_agent(),
    )


@app.get("/health", response_model=HealthResponse)
def health(qualitative_agent: QualitativeAgent = Depends(get_qualitative_agent)) -> HealthResponse:
    settings = get_settings()

    chroma_ok = True
    try:
        qualitative_agent.collection.count()
    except Exception:
        chroma_ok = False

    sqlite_ok = True
    try:
        conn = sqlite3.connect(settings.sqlite_path)
        conn.execute("SELECT 1 FROM customers LIMIT 1")
        conn.close()
    except Exception:
        sqlite_ok = False

    status = "ok" if chroma_ok and sqlite_ok else "degraded"
    return HealthResponse(status=status, chroma_ok=chroma_ok, sqlite_ok=sqlite_ok)


@app.post("/query", response_model=QueryResponse)
def query(
    request_body: QueryRequest, manager_agent: ManagerAgent = Depends(get_manager_agent)
) -> QueryResponse:
    result = manager_agent.handle(request_body.query)
    return QueryResponse(
        final_answer=result.final_answer,
        parts=[AnswerPartSchema(agent=p.agent, answer=p.answer) for p in result.parts],
        clarification_needed=result.clarification_needed,
        clarification_question=result.clarification_question,
    )


@app.post("/qualitative", response_model=QualitativeResponse)
def qualitative(
    request_body: QueryRequest,
    qualitative_agent: QualitativeAgent = Depends(get_qualitative_agent),
) -> QualitativeResponse:
    result = qualitative_agent.answer(request_body.query)
    return QualitativeResponse(
        answer=result.answer,
        citations=[
            CitationSchema(doc_id=c.doc_id, similarity=c.similarity, snippet=c.snippet)
            for c in result.citations
        ],
        used_fallback=result.used_fallback,
    )


@app.post("/quantitative", response_model=QuantitativeResponse)
def quantitative(
    request_body: QueryRequest,
    quantitative_agent: QuantitativeAgent = Depends(get_quantitative_agent),
) -> QuantitativeResponse:
    result = quantitative_agent.answer(request_body.query)
    return QuantitativeResponse(
        answer=result.answer, sql=result.sql, rows=result.rows, chart_path=result.chart_path
    )
