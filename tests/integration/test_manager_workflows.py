"""End-to-end Manager workflows against real Chroma + real SQLite, with
only the Gemini calls mocked (classification, RAG generation, NL-to-SQL,
summarization)."""

from config import Settings
from agents.manager_agent import ManagerAgent
from agents.qualitative_agent import QualitativeAgent
from agents.quantitative_agent import QuantitativeAgent


def _full_settings(ingested_settings: Settings, seeded_db: Settings) -> Settings:
    """Merge the pre-ingested Chroma path with the seeded SQLite path so a
    single settings object has both real backends available."""
    return Settings(**{**ingested_settings.__dict__, "sqlite_path": seeded_db.sqlite_path})


def _build_manager(settings: Settings, mock_gemini_client):
    qualitative_agent = QualitativeAgent(settings=settings, gemini_client=mock_gemini_client)
    quantitative_agent = QuantitativeAgent(settings=settings, gemini_client=mock_gemini_client)
    return ManagerAgent(
        settings=settings,
        gemini_client=mock_gemini_client,
        qualitative_agent=qualitative_agent,
        quantitative_agent=quantitative_agent,
    )


def test_qualitative_only_workflow_end_to_end(ingested_settings, seeded_db, mock_gemini_client):
    settings = _full_settings(ingested_settings, seeded_db)

    mock_gemini_client.generate.side_effect = [
        "qualitative",  # classification
        "Per the security policy, MFA is required.",  # RAG answer
    ]
    manager = _build_manager(settings, mock_gemini_client)

    result = manager.handle("What does our security policy say about MFA?")

    assert len(result.parts) == 1
    assert result.parts[0].agent == "qualitative"
    assert "MFA" in result.final_answer


def test_quantitative_only_workflow_end_to_end(ingested_settings, seeded_db, mock_gemini_client):
    settings = _full_settings(ingested_settings, seeded_db)

    mock_gemini_client.generate.side_effect = [
        "quantitative",  # classification
        "SELECT COUNT(*) AS customer_count FROM customers",  # generated SQL
        "There are 40 customers in total.",  # summarization
    ]
    manager = _build_manager(settings, mock_gemini_client)

    result = manager.handle("How many customers do we have in total?")

    assert len(result.parts) == 1
    assert result.parts[0].agent == "quantitative"
    assert "40 customers" in result.final_answer


def test_both_agents_workflow_end_to_end(ingested_settings, seeded_db, mock_gemini_client):
    settings = _full_settings(ingested_settings, seeded_db)

    mock_gemini_client.generate.side_effect = [
        "both",  # classification
        "Employee satisfaction is discussed in our HR policy docs.",  # qualitative answer
        "SELECT COUNT(*) AS customer_count FROM customers",  # generated SQL
        "There are 40 customers.",  # quantitative summarization
    ]
    manager = _build_manager(settings, mock_gemini_client)

    result = manager.handle(
        "How does employee satisfaction compare to our customer count, and what policies apply?"
    )

    assert len(result.parts) == 2
    agents_used = {part.agent for part in result.parts}
    assert agents_used == {"qualitative", "quantitative"}
    assert "[Qualitative Agent]" in result.final_answer
    assert "[Quantitative Agent]" in result.final_answer
