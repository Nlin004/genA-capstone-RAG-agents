from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agents.manager_agent import AnswerPart, ManagerResult
from agents.qualitative_agent import Citation, QualitativeResult
from agents.quantitative_agent import QuantitativeResult
from api.main import app, get_manager_agent, get_qualitative_agent, get_quantitative_agent


@pytest.fixture
def client():
    mock_manager = MagicMock()
    mock_qualitative = MagicMock()
    mock_quantitative = MagicMock()

    app.dependency_overrides[get_manager_agent] = lambda: mock_manager
    app.dependency_overrides[get_qualitative_agent] = lambda: mock_qualitative
    app.dependency_overrides[get_quantitative_agent] = lambda: mock_quantitative

    with TestClient(app) as test_client:
        yield test_client, mock_manager, mock_qualitative, mock_quantitative

    app.dependency_overrides.clear()


def test_health_endpoint_reports_ok_when_both_backends_reachable(client):
    test_client, _, mock_qualitative, _ = client
    mock_qualitative.collection.count.return_value = 42

    response = test_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "chroma_ok" in body and "sqlite_ok" in body


def test_query_endpoint_returns_manager_result(client):
    test_client, mock_manager, _, _ = client
    mock_manager.handle.return_value = ManagerResult(
        final_answer="[Qualitative Agent]\nanswer text",
        parts=[AnswerPart(agent="qualitative", answer="answer text")],
    )

    response = test_client.post("/query", json={"query": "What is our security policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"] == "[Qualitative Agent]\nanswer text"
    assert body["parts"] == [{"agent": "qualitative", "answer": "answer text"}]
    assert body["clarification_needed"] is False
    mock_manager.handle.assert_called_once_with("What is our security policy?")


def test_query_endpoint_rejects_empty_query(client):
    test_client, _, _, _ = client

    response = test_client.post("/query", json={"query": ""})

    assert response.status_code == 422


def test_qualitative_endpoint_returns_citations(client):
    test_client, _, mock_qualitative, _ = client
    mock_qualitative.answer.return_value = QualitativeResult(
        answer="policy answer",
        citations=[Citation(doc_id="security_policy", similarity=0.9, snippet="...")],
    )

    response = test_client.post("/qualitative", json={"query": "What is the security policy?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "policy answer"
    assert body["citations"][0]["doc_id"] == "security_policy"


def test_quantitative_endpoint_returns_sql_and_rows(client):
    test_client, _, _, mock_quantitative = client
    mock_quantitative.answer.return_value = QuantitativeResult(
        answer="There are 40 customers.",
        sql="SELECT COUNT(*) AS customer_count FROM customers",
        rows=[{"customer_count": 40}],
    )

    response = test_client.post("/quantitative", json={"query": "How many customers?"})

    assert response.status_code == 200
    body = response.json()
    assert body["sql"] == "SELECT COUNT(*) AS customer_count FROM customers"
    assert body["rows"] == [{"customer_count": 40}]
