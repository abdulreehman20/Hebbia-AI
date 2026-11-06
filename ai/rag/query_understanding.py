"""Classify and extract structured intent from investor questions."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ai.providers.openrouter_chat import get_chat_model
from ai.rag.namespace_registry import SUPPORTED_TICKERS, validate_tickers
from ai.schemas.research import ResearchIntent, ResearchRequest, TaskType
from config import Settings, get_settings

logger = logging.getLogger(__name__)

TICKER_ALIASES: dict[str, str] = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "META": "META",
    "FACEBOOK": "META",
    "WALMART": "WMT",
    "ORACLE": "ORCL",
}

TASK_KEYWORDS: list[tuple[TaskType, tuple[str, ...]]] = [
    ("company_comparison", ("compare", "versus", " vs ", "relative to")),
    ("financial_calculations", ("margin", "growth rate", "calculate", "ratio", "free cash flow")),
    ("risk_analysis", ("risk factor", "risks", "uncertainty")),
    ("management_discussion", ("md&a", "management discussion", "item 7")),
    ("corporate_events", ("8-k", "acquisition", "merger", "dividend", "buyback")),
    ("cash_flow_and_capital", ("cash flow", "capex", "capital expenditure", "liquidity")),
    ("business_segment_analysis", ("segment", "product line", "geography")),
    ("accounting_and_footnotes", ("footnote", "accounting policy", "note ")),
    ("revenue_and_profitability", ("revenue", "profit", "income", "earnings", "margin")),
    ("historical_trends", ("trend", "over time", "year over year", "historical")),
    ("financial_statement_analysis", ("balance sheet", "income statement", "10-k", "10-q")),
    ("investment_due_diligence", ("due diligence", "investment thesis", "should i invest")),
    ("company_overview", ("overview", "about the company", "what does", "business description")),
]


class QueryUnderstandingService:
    """Produce a validated ResearchIntent from a ResearchRequest."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def understand(self, request: ResearchRequest) -> ResearchIntent:
        heuristic = self._heuristic_intent(request)
        try:
            llm_intent = self._llm_intent(request)
            return self._merge(heuristic, llm_intent, request)
        except Exception as exc:  # noqa: BLE001 — fall back to heuristics
            logger.warning("LLM query understanding failed; using heuristics: %s", exc)
            return heuristic

    def _heuristic_intent(self, request: ResearchRequest) -> ResearchIntent:
        question = request.question
        lower = question.lower()
        tickers = list(request.selected_companies)
        tickers.extend(self._extract_tickers(question))
        try:
            tickers = validate_tickers(tickers) if tickers else []
        except ValueError:
            tickers = [t for t in tickers if t in SUPPORTED_TICKERS]

        task_type: TaskType = "unsupported"
        for candidate, keywords in TASK_KEYWORDS:
            if any(keyword in lower for keyword in keywords):
                task_type = candidate
                break
        if task_type == "unsupported" and tickers:
            task_type = "company_overview"

        years = [int(y) for y in re.findall(r"\b(20[1-2]\d)\b", question)]
        start_year = min(years) if years else None
        end_year = max(years) if years else None

        metrics: list[str] = []
        for metric in (
            "revenue",
            "gross_margin",
            "operating_margin",
            "net_profit_margin",
            "free_cash_flow",
            "debt_to_equity",
            "operating_income",
            "net_income",
        ):
            token = metric.replace("_", " ")
            if token in lower or metric in lower:
                metrics.append(metric)

        requires_calc = task_type == "financial_calculations" or any(
            m in metrics
            for m in (
                "gross_margin",
                "operating_margin",
                "net_profit_margin",
                "free_cash_flow",
                "debt_to_equity",
                "revenue_growth",
            )
        )
        if "growth" in lower and "revenue" in lower and "revenue_growth" not in metrics:
            metrics.append("revenue_growth")
            requires_calc = True

        filing_types = list(request.filing_types)
        for form in ("10-K", "10-Q", "8-K"):
            if form.lower() in lower and form not in filing_types:
                filing_types.append(form)

        unsupported_reason = None
        if task_type == "unsupported":
            unsupported_reason = (
                "Question is outside the supported financial research corpus "
                "or does not reference an allowed company."
            )

        return ResearchIntent(
            task_type=task_type,
            tickers=tickers,
            metrics=metrics,
            start_year=start_year,
            end_year=end_year,
            filing_types=filing_types,
            sections=[],
            requires_calculation=requires_calc,
            requires_multiple_namespaces=len(tickers) > 1,
            search_queries=[question.strip()],
            unsupported_reason=unsupported_reason,
        )

    def _llm_intent(self, request: ResearchRequest) -> ResearchIntent:
        model = get_chat_model(self.settings, temperature=0.0)
        system = SystemMessage(
            content=(
                "Classify the investor question for a SEC-filing research system. "
                f"Allowed tickers only: {sorted(SUPPORTED_TICKERS)}. "
                "Return JSON matching ResearchIntent fields: task_type, tickers, "
                "metrics, start_year, end_year, filing_types, sections, "
                "requires_calculation, requires_multiple_namespaces, search_queries, "
                "unsupported_reason. Do not invent unsupported tickers."
            )
        )
        human = HumanMessage(
            content=json.dumps(
                {
                    "question": request.question,
                    "selected_companies": request.selected_companies,
                    "filing_types": request.filing_types,
                }
            )
        )
        structured = model.with_structured_output(ResearchIntent)
        result = structured.invoke([system, human])
        if isinstance(result, ResearchIntent):
            return result
        if isinstance(result, dict):
            return ResearchIntent.model_validate(result)
        raise TypeError(f"Unexpected structured output type: {type(result)}")

    def _merge(
        self,
        heuristic: ResearchIntent,
        llm: ResearchIntent,
        request: ResearchRequest,
    ) -> ResearchIntent:
        tickers = llm.tickers or heuristic.tickers or list(request.selected_companies)
        tickers = validate_tickers(tickers) if tickers else []
        task_type = llm.task_type if llm.task_type != "unsupported" else heuristic.task_type
        if not tickers and task_type != "unsupported":
            task_type = "unsupported"
        return ResearchIntent(
            task_type=task_type,
            tickers=tickers,
            metrics=llm.metrics or heuristic.metrics,
            start_year=llm.start_year or heuristic.start_year,
            end_year=llm.end_year or heuristic.end_year,
            filing_types=llm.filing_types or heuristic.filing_types,
            sections=llm.sections or heuristic.sections,
            requires_calculation=llm.requires_calculation or heuristic.requires_calculation,
            requires_multiple_namespaces=len(tickers) > 1,
            search_queries=llm.search_queries or heuristic.search_queries,
            unsupported_reason=llm.unsupported_reason if task_type == "unsupported" else None,
        )

    @staticmethod
    def _extract_tickers(question: str) -> list[str]:
        found: list[str] = []
        upper = question.upper()
        for ticker in SUPPORTED_TICKERS:
            if re.search(rf"\b{re.escape(ticker)}\b", upper):
                found.append(ticker)
        for alias, ticker in TICKER_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", upper) and ticker not in found:
                found.append(ticker)
        return found
