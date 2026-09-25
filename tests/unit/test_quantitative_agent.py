import pytest

from agents.quantitative_agent import (
    QuantitativeAgent,
    SqlValidationError,
    validate_select_only,
)


class TestValidateSelectOnly:
    def test_accepts_plain_select(self):
        assert validate_select_only("SELECT * FROM customers") == "SELECT * FROM customers"

    def test_accepts_trailing_semicolon(self):
        assert validate_select_only("SELECT 1;") == "SELECT 1"

    def test_rejects_non_select(self):
        with pytest.raises(SqlValidationError):
            validate_select_only("DELETE FROM customers")

    def test_rejects_multiple_statements(self):
        with pytest.raises(SqlValidationError):
            validate_select_only("SELECT 1; DROP TABLE customers;")

    def test_rejects_banned_keyword_inside_select(self):
        with pytest.raises(SqlValidationError):
            validate_select_only("SELECT * FROM customers WHERE 1=1; UPDATE customers SET name='x'")

    def test_rejects_empty_sql(self):
        with pytest.raises(SqlValidationError):
            validate_select_only("   ")


def test_answer_executes_generated_sql_against_real_db(seeded_db, mock_gemini_client):
    mock_gemini_client.generate.side_effect = [
        "SELECT COUNT(*) AS customer_count FROM customers",
        "There are 40 customers.",
    ]
    agent = QuantitativeAgent(settings=seeded_db, gemini_client=mock_gemini_client)

    result = agent.answer("How many customers do we have?")

    assert result.sql == "SELECT COUNT(*) AS customer_count FROM customers"
    assert result.rows == [{"customer_count": 40}]
    assert result.answer == "There are 40 customers."
    assert mock_gemini_client.generate.call_count == 2


def test_answer_rejects_unsafe_generated_sql_without_executing(seeded_db, mock_gemini_client):
    mock_gemini_client.generate.return_value = "DROP TABLE customers"
    agent = QuantitativeAgent(settings=seeded_db, gemini_client=mock_gemini_client)

    result = agent.answer("delete everyone")

    assert result.sql is None
    assert "safe SQL" in result.answer
    mock_gemini_client.generate.assert_called_once()


def test_answer_renders_chart_for_trend_shaped_results(seeded_db, mock_gemini_client):
    mock_gemini_client.generate.side_effect = [
        "SELECT r.name AS region, COUNT(*) AS customer_count "
        "FROM customers c JOIN regions r ON r.id = c.region_id GROUP BY r.name",
        "Customer counts vary by region.",
    ]
    agent = QuantitativeAgent(settings=seeded_db, gemini_client=mock_gemini_client)

    result = agent.answer("Compare customer counts across regions")

    assert result.chart_path is not None
    assert result.chart_path.endswith(".png")
    from pathlib import Path

    assert Path(result.chart_path).exists()
