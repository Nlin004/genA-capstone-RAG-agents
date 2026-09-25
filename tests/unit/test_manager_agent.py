from unittest.mock import MagicMock

from agents.manager_agent import CLARIFICATION_QUESTION, ManagerAgent
from agents.qualitative_agent import QualitativeResult
from agents.quantitative_agent import QuantitativeResult


def _make_manager(mock_gemini_client, classification: str):
    mock_gemini_client.generate.return_value = classification
    qualitative_agent = MagicMock()
    qualitative_agent.answer.return_value = QualitativeResult(answer="qual answer", citations=[])
    quantitative_agent = MagicMock()
    quantitative_agent.answer.return_value = QuantitativeResult(answer="quant answer", sql="SELECT 1")

    manager = ManagerAgent(
        gemini_client=mock_gemini_client,
        qualitative_agent=qualitative_agent,
        quantitative_agent=quantitative_agent,
    )
    return manager, qualitative_agent, quantitative_agent


def test_routes_qualitative_only(mock_gemini_client):
    manager, qualitative_agent, quantitative_agent = _make_manager(mock_gemini_client, "qualitative")

    result = manager.handle("What is our security policy?")

    qualitative_agent.answer.assert_called_once()
    quantitative_agent.answer.assert_not_called()
    assert len(result.parts) == 1
    assert result.parts[0].agent == "qualitative"
    assert "qual answer" in result.final_answer
    assert result.clarification_needed is False


def test_routes_quantitative_only(mock_gemini_client):
    manager, qualitative_agent, quantitative_agent = _make_manager(mock_gemini_client, "quantitative")

    result = manager.handle("What's our revenue trend?")

    quantitative_agent.answer.assert_called_once()
    qualitative_agent.answer.assert_not_called()
    assert len(result.parts) == 1
    assert result.parts[0].agent == "quantitative"


def test_routes_both_and_merges_labeled_parts(mock_gemini_client):
    manager, qualitative_agent, quantitative_agent = _make_manager(mock_gemini_client, "both")

    result = manager.handle("How does satisfaction compare to policy impact?")

    qualitative_agent.answer.assert_called_once()
    quantitative_agent.answer.assert_called_once()
    assert len(result.parts) == 2
    assert "[Qualitative Agent]" in result.final_answer
    assert "[Quantitative Agent]" in result.final_answer


def test_ambiguous_classification_asks_clarifying_question(mock_gemini_client):
    manager, qualitative_agent, quantitative_agent = _make_manager(mock_gemini_client, "ambiguous")

    result = manager.handle("tell me about it")

    qualitative_agent.answer.assert_not_called()
    quantitative_agent.answer.assert_not_called()
    assert result.clarification_needed is True
    assert result.final_answer == CLARIFICATION_QUESTION


def test_unrecognized_label_falls_back_to_ambiguous(mock_gemini_client):
    manager, qualitative_agent, quantitative_agent = _make_manager(mock_gemini_client, "banana")

    result = manager.handle("something weird")

    assert result.clarification_needed is True
    qualitative_agent.answer.assert_not_called()
    quantitative_agent.answer.assert_not_called()
