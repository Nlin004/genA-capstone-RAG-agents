from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from agents.qualitative_agent import QualitativeAgent
from agents.quantitative_agent import QuantitativeAgent
from config import Settings, get_settings
from llm.gemini_client import GeminiClient, GeminiError
from logging_config import get_logger

logger = get_logger(__name__)

VALID_CLASSIFICATIONS = {"qualitative", "quantitative", "both", "ambiguous"}

# --- Rule-based fallback signals!!!!! --------------------------------------------
# This is ONLY consulted if the LLM classification call fails or returns something
# outside VALID_CLASSIFICATIONS. Phrases first (more specific), then single
# tokens. \b keeps short words from matching inside other words (e.g. "sum"
# in "summary").  This was implemented bc I was getting quota hits super fast on the free
# gemini plan and didn't want to spend additional calls deciding which agent to use.
_QUANTITATIVE_TERMS = [
    r"\bhow many\b", r"\bhow much\b", r"\bhow often\b",
    r"\brevenue\b", r"\bchurn\b", r"\bsales\b", r"\bprofit\b", r"\bmargin\b",
    r"\btrend(s)?\b", r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bcount\b", r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bavg\b", r"\bmean\b",
    r"\brate\b", r"\bratio\b", r"\bpercent(age)?\b", r"%",
    r"\bmetric(s)?\b", r"\bnumber(s)?\b", r"\bfigure(s)?\b", r"\bstatistic(s)?\b",
    r"\bgrowth\b", r"\bdecline\b", r"\btop\b", r"\bhighest\b", r"\blowest\b",
    r"\bper (month|quarter|region|year|customer)\b",
    r"\bmonthly\b", r"\bquarterly\b", r"\byearly\b", r"\bannual\b",
    r"\bq[1-4]\b", r"\bquarter\b", r"\bregion(s|al)?\b",
    r"\bbreakdown\b", r"\bdistribution\b",
    r"\bmrr\b", r"\barr\b", r"\bnps\b", r"\bkpi(s)?\b",
    r"\bconversion\b", r"\bretention\b", r"\bforecast\b", r"\bdashboard\b",
]
_QUALITATIVE_TERMS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bwhat is (our|the)\b", r"\bwhat's (our|the)\b",
    r"\bhow do (we|i|you)\b", r"\bhow does (our|the)\b", r"\bhow to\b",
    r"\bpolicy\b", r"\bpolicies\b", r"\bprocess(es)?\b", r"\bprocedure(s)?\b",
    r"\bguideline(s)?\b", r"\bdocumentation\b", r"\bdocument(s)?\b",
    r"\bhandle\b", r"\bhandling\b", r"\breview\b", r"\bonboard(ing)?\b",
    r"\bsecurity\b", r"\bcompliance\b", r"\bgovernance\b",
    r"\bstrategy\b", r"\bapproach\b", r"\bstandard(s)?\b",
    r"\bbest practice(s)?\b", r"\bcomplaint(s)?\b", r"\bescalation(s)?\b",
    r"\btraining\b", r"\bsop\b",
]
# Deliberately dropped from the qualitative list: bare "why" and "what
# should" are too generic and would flag plenty of pure-quantitative
# questions ("why did churn increase this quarter") as "both", triggering
# an extra agent call the fallback is specifically trying to avoid.

_QUANTITATIVE_PATTERNS = [re.compile(t, re.IGNORECASE) for t in _QUANTITATIVE_TERMS]
_QUALITATIVE_PATTERNS = [re.compile(t, re.IGNORECASE) for t in _QUALITATIVE_TERMS]

CLASSIFICATION_PROMPT = """\
Classify the user's question into exactly one label:
- "qualitative": about policies, processes, documentation, or how something works.
- "quantitative": about numbers, metrics, revenue, counts, trends, or comparisons from data.
- "both": the question clearly needs both a data answer AND policy/process context.
- "ambiguous": too vague or unclear to route confidently.

Respond with ONLY the single label word, nothing else.

QUESTION: {query}
"""

CLARIFICATION_QUESTION = (
    "I'm not sure whether you're asking about a policy/process topic or "
    "about data and numbers. Could you rephrase or add more detail?"
)


@dataclass
class AnswerPart:
    agent: str
    answer: str


@dataclass
class ManagerResult:
    final_answer: str
    parts: list[AnswerPart] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: str | None = None


def _count_hits(query: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(1 for pat in patterns if pat.search(query))


class ManagerAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        gemini_client: GeminiClient | None = None,
        qualitative_agent: QualitativeAgent | None = None,
        quantitative_agent: QuantitativeAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gemini_client = gemini_client or GeminiClient(self.settings)
        self.qualitative_agent = qualitative_agent or QualitativeAgent(
            self.settings, self.gemini_client
        )
        self.quantitative_agent = quantitative_agent or QuantitativeAgent(
            self.settings, self.gemini_client
        )

    def _classify_by_llm(self, query: str) -> str | None:
        """Normal path. Returns a valid label, or None if the call failed
        or came back with something outside VALID_CLASSIFICATIONS."""
        try:
            raw = self.gemini_client.generate(CLASSIFICATION_PROMPT.format(query=query))
        except GeminiError as exc:
            logger.warning("manager_classify_llm_unavailable", extra={"error": str(exc)})
            return None

        label = raw.strip().strip('"').lower()
        return label if label in VALID_CLASSIFICATIONS else None

    def _classify_by_rules(self, query: str) -> str | None:
        """Manual fallback, only reached when the LLM path returns None.
        Returns a label from keyword hits, or None if there's no signal
        at all."""
        quant = _count_hits(query, _QUANTITATIVE_PATTERNS)
        qual = _count_hits(query, _QUALITATIVE_PATTERNS)

        if quant and qual:
            return "both"
        if quant:
            return "quantitative"
        if qual:
            return "qualitative"
        return None

    def _classify(self, query: str) -> str:
        llm_label = self._classify_by_llm(query)
        if llm_label is not None:
            return llm_label

        logger.info("manager_classify_falling_back_to_rules", extra={"query": query})
        rule_label = self._classify_by_rules(query)
        if rule_label is not None:
            return rule_label

        # Neither the LLM nor the keyword rules found anything - ask rather
        # than guess.
        return "ambiguous"

    def handle(self, query: str) -> ManagerResult:
        start = time.monotonic()
        classification = self._classify(query)

        if classification == "ambiguous":
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(
                "manager_ambiguous_query",
                extra={"query": query, "selected_agent": "none", "elapsed_ms": elapsed_ms},
            )
            return ManagerResult(
                final_answer=CLARIFICATION_QUESTION,
                clarification_needed=True,
                clarification_question=CLARIFICATION_QUESTION,
            )

        parts: list[AnswerPart] = []

        if classification in ("qualitative", "both"):
            qual_result = self.qualitative_agent.answer(query)
            parts.append(AnswerPart(agent="qualitative", answer=qual_result.answer))

        if classification in ("quantitative", "both"):
            quant_result = self.quantitative_agent.answer(query)
            parts.append(AnswerPart(agent="quantitative", answer=quant_result.answer))

        final_answer = "\n\n".join(f"[{part.agent.title()} Agent]\n{part.answer}" for part in parts)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "manager_query_handled",
            extra={
                "query": query,
                "selected_agent": classification,
                "elapsed_ms": elapsed_ms,
            },
        )
        return ManagerResult(final_answer=final_answer, parts=parts)
