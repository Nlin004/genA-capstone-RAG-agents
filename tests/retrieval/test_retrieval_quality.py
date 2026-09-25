"""Retrieval-quality evaluation: for each doc, a question we'd expect it
to answer should retrieve that doc above the relevance threshold, and a
question with no answer in the knowledge base should retrieve nothing."""

import pytest

from agents.qualitative_agent import NOT_FOUND_MESSAGE, QualitativeAgent

EXPECTED_DOC_FOR_QUESTION = [
    ("What are the password requirements for employees?", "security_policy"),
    ("How many approvals does a pull request need before merging?", "code_review_process"),
    ("What are the severity levels for customer complaints?", "customer_complaints"),
    ("What was the overall employee engagement score in the latest survey?", "employee_satisfaction"),
    ("How are Enterprise customer accounts assigned a CS manager?", "customer_success_strategy"),
]

OUT_OF_KB_QUESTIONS = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]


@pytest.mark.parametrize("question,expected_doc_id", EXPECTED_DOC_FOR_QUESTION)
def test_expected_document_is_retrieved_above_threshold(
    ingested_settings, mock_gemini_client, question, expected_doc_id
):
    agent = QualitativeAgent(settings=ingested_settings, gemini_client=mock_gemini_client)

    result = agent.answer(question)

    assert result.used_fallback is False
    retrieved_doc_ids = {c.doc_id for c in result.citations}
    assert expected_doc_id in retrieved_doc_ids

    top_citation = max(result.citations, key=lambda c: c.similarity)
    assert top_citation.similarity >= ingested_settings.retrieval_similarity_threshold


@pytest.mark.parametrize("question", OUT_OF_KB_QUESTIONS)
def test_out_of_kb_questions_return_not_found_fallback(ingested_settings, mock_gemini_client, question):
    agent = QualitativeAgent(settings=ingested_settings, gemini_client=mock_gemini_client)

    result = agent.answer(question)

    assert result.used_fallback is True
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    mock_gemini_client.generate.assert_not_called()
