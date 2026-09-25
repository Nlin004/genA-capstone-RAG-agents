"""Quantitative NL-to-SQL Agent: translates a question into a read-only SQL
query against the dummy SQLite database, executes it, summarizes the
result, and optionally renders a chart."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt

from config import Settings, get_settings
from llm.gemini_client import GeminiClient, GeminiError
from logging_config import get_logger

logger = get_logger(__name__)

# Kept in sync with scripts/setup_dummy_db.py's SCHEMA. Hardcoded rather than
# introspected because it doubles as the prompt shown to the LLM, and the
# schema only changes when that script changes.
SCHEMA_DESCRIPTION = """\
regions(id INTEGER, name TEXT)
customers(id INTEGER, name TEXT, region_id INTEGER, signup_date TEXT, churned_at TEXT NULL)
subscriptions(id INTEGER, customer_id INTEGER, plan TEXT, mrr REAL, start_date TEXT, end_date TEXT NULL)
invoices(id INTEGER, customer_id INTEGER, amount REAL, invoice_date TEXT)
"""

_BANNED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|TRUNCATE|VACUUM)\b",
    re.IGNORECASE,
)


class SqlValidationError(ValueError):
    pass


@dataclass
class QuantitativeResult:
    answer: str
    sql: str | None = None
    rows: list[dict] = field(default_factory=list)
    chart_path: str | None = None


def validate_select_only(sql: str) -> str:
    """Raises SqlValidationError unless sql is a single, read-only SELECT."""
    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise SqlValidationError("Empty SQL.")
    if ";" in cleaned:
        raise SqlValidationError("Multiple statements are not allowed.")
    if not cleaned.upper().startswith("SELECT"):
        raise SqlValidationError("Only SELECT statements are allowed.")
    if _BANNED_KEYWORDS.search(cleaned):
        raise SqlValidationError("SQL contains a disallowed keyword.")

    return cleaned


def _looks_like_trend(rows: list[dict]) -> bool:
    if len(rows) < 2:
        return False
    columns = list(rows[0].keys())
    if len(columns) < 2:
        return False
    numeric_columns = [c for c in columns if isinstance(rows[0][c], (int, float))]
    return len(numeric_columns) >= 1 and len(columns) - len(numeric_columns) >= 1


def _render_chart(rows: list[dict], chart_dir: Path) -> str:
    columns = list(rows[0].keys())
    numeric_col = next(c for c in columns if isinstance(rows[0][c], (int, float)))
    label_col = next(c for c in columns if c != numeric_col)

    labels = [str(row[label_col]) for row in rows]
    values = [row[numeric_col] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_xlabel(label_col)
    ax.set_ylabel(numeric_col)
    ax.set_title(f"{numeric_col} by {label_col}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    chart_dir.mkdir(parents=True, exist_ok=True)
    filename = f"chart_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.png"
    chart_path = chart_dir / filename
    fig.savefig(chart_path)
    plt.close(fig)

    return str(chart_path)


class QuantitativeAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        gemini_client: GeminiClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gemini_client = gemini_client or GeminiClient(self.settings)

    def _generate_sql(self, query: str) -> str:
        prompt = (
            "You translate natural-language questions into a single SQLite "
            "SELECT statement. Use ONLY the tables/columns below. Return ONLY "
            "the SQL, no explanation, no markdown fences.\n\n"
            f"SCHEMA:\n{SCHEMA_DESCRIPTION}\n\nQUESTION: {query}"
        )
        return self.gemini_client.generate(prompt)

    def _run_sql(self, sql: str) -> list[dict]:
        conn = sqlite3.connect(self.settings.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _summarize(self, query: str, rows: list[dict]) -> str:
        preview = rows[:20]
        prompt = (
            f"The question was: {query}\n"
            f"The SQL query returned these rows (JSON): {preview}\n"
            "Write a concise, plain-language answer to the question based on "
            "these results. Mention concrete numbers where relevant."
        )
        return self.gemini_client.generate(prompt)

    def answer(self, query: str) -> QuantitativeResult:
        start = time.monotonic()

        try:
            raw_sql = self._generate_sql(query)
            sql = validate_select_only(raw_sql)
        except (GeminiError, SqlValidationError) as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.error(
                "quantitative_sql_generation_failed",
                extra={"query": query, "elapsed_ms": elapsed_ms, "error": str(exc)},
            )
            return QuantitativeResult(
                answer=f"I couldn't build a safe SQL query for that question ({exc})."
            )

        try:
            rows = self._run_sql(sql)
        except sqlite3.Error as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.error(
                "quantitative_sql_execution_failed",
                extra={"query": query, "sql": sql, "elapsed_ms": elapsed_ms, "error": str(exc)},
            )
            return QuantitativeResult(
                answer=f"The generated query failed to execute: {exc}", sql=sql
            )

        chart_path = None
        if rows and _looks_like_trend(rows):
            try:
                chart_path = _render_chart(rows, self.settings.chart_output_dir)
            except Exception as exc:  # noqa: BLE001 - charting is best-effort, never blocks the answer
                logger.warning(
                    "quantitative_chart_failed",
                    extra={"query": query, "sql": sql, "error": str(exc)},
                )

        try:
            summary = self._summarize(query, rows) if rows else "No matching rows were found."
        except GeminiError:
            summary = f"Query executed successfully and returned {len(rows)} row(s)."

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "quantitative_answer_success",
            extra={"query": query, "sql": sql, "elapsed_ms": elapsed_ms, "row_count": len(rows)},
        )
        return QuantitativeResult(answer=summary, sql=sql, rows=rows, chart_path=chart_path)
