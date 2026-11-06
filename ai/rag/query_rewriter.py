"""Transform investor questions into focused retrieval queries."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.providers.openrouter_chat import get_chat_model
from ai.schemas.research import ResearchIntent
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    queries: list[str] = Field(default_factory=list)


class QueryRewriter:
    """Generate a bounded set of focused search queries."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rewrite(self, question: str, intent: ResearchIntent) -> list[str]:
        limit = max(1, self.settings.max_query_rewrites)
        seed = self._heuristic_queries(question, intent)[:limit]
        try:
            llm_queries = self._llm_queries(question, intent, limit)
            merged = self._dedupe(seed + llm_queries)[:limit]
            return merged or [question.strip()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite LLM failed; using heuristics: %s", exc)
            return seed or [question.strip()]

    def _heuristic_queries(self, question: str, intent: ResearchIntent) -> list[str]:
        queries: list[str] = [question.strip()]
        period = ""
        if intent.start_year and intent.end_year:
            period = f" fiscal years {intent.start_year} to {intent.end_year}"
        elif intent.start_year:
            period = f" fiscal year {intent.start_year}"

        metrics = intent.metrics or ["financial performance"]
        for ticker in intent.tickers:
            for metric in metrics[:3]:
                label = metric.replace("_", " ")
                queries.append(f"{ticker} {label}{period}".strip())

        if intent.task_type == "risk_analysis":
            for ticker in intent.tickers:
                queries.append(f"{ticker} Item 1A Risk Factors{period}".strip())
        if intent.task_type in {"management_discussion", "revenue_and_profitability"}:
            for ticker in intent.tickers:
                queries.append(f"{ticker} Item 7 MD&A revenue operating income{period}".strip())
        if intent.requires_multiple_namespaces and len(intent.tickers) >= 2:
            joined = " and ".join(intent.tickers)
            queries.append(f"{joined} comparative financial performance{period}".strip())
        return self._dedupe(queries)

    def _llm_queries(self, question: str, intent: ResearchIntent, limit: int) -> list[str]:
        model = get_chat_model(self.settings, temperature=0.0)
        structured = model.with_structured_output(RewriteResult)
        system = SystemMessage(
            content=(
                "Rewrite the investor question into focused SEC-filing search queries. "
                f"Return at most {limit} queries. Do not invent facts or unsupported companies. "
                "Preserve ticker and period constraints."
            )
        )
        human = HumanMessage(
            content=(
                f"Question: {question}\n"
                f"Intent JSON: {intent.model_dump_json()}\n"
                f"Max queries: {limit}"
            )
        )
        result = structured.invoke([system, human])
        if isinstance(result, RewriteResult):
            return [q.strip() for q in result.queries if q.strip()]
        if isinstance(result, dict):
            return [q.strip() for q in result.get("queries", []) if isinstance(q, str) and q.strip()]
        return []

    @staticmethod
    def _dedupe(queries: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for query in queries:
            key = query.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(query.strip())
        return out
