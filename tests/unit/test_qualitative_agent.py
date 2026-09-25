from agents.qualitative_agent import NOT_FOUND_MESSAGE, QualitativeAgent


def _make_agent(ingested_settings, mock_gemini_client):
    return QualitativeAgent(settings=ingested_settings, gemini_client=mock_gemini_client)


def test_relevant_query_returns_citations_and_calls_llm(ingested_settings, mock_gemini_client):
    agent = _make_agent(ingested_settings, mock_gemini_client)

    result = agent.answer("What is our company's security policy on passwords?")

    assert result.used_fallback is False
    assert result.answer == "mocked answer"
    assert len(result.citations) > 0
    assert any(c.doc_id == "security_policy" for c in result.citations)
    mock_gemini_client.generate.assert_called_once()


def test_out_of_kb_query_short_circuits_without_calling_llm(ingested_settings, mock_gemini_client):
    agent = _make_agent(ingested_settings, mock_gemini_client)

    result = agent.answer("What is the airspeed velocity of an unladen swallow?")

    assert result.used_fallback is True
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []
    mock_gemini_client.generate.assert_not_called()


def test_citations_carry_similarity_scores_above_threshold(ingested_settings, mock_gemini_client):
    agent = _make_agent(ingested_settings, mock_gemini_client)

    result = agent.answer("Explain the code review process for pull requests")

    assert result.used_fallback is False
    for citation in result.citations:
        assert citation.similarity >= ingested_settings.retrieval_similarity_threshold
