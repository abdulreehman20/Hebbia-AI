"""Financial analysis and calculation tools."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai.calculations.engine import CalculationEngine
from ai.calculations.financial_facts import FinancialFactExtractor
from ai.schemas.retrieval import RetrievedChunk


class _ExtractArgs(BaseModel):
    chunks: list[dict[str, Any]]
    metrics: list[str] = Field(default_factory=list)


@tool("extract_financial_facts", args_schema=_ExtractArgs)
def extract_financial_facts(
    chunks: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Extract structured numerical facts from retrieved chunk payloads."""
    parsed = [RetrievedChunk.model_validate(chunk) for chunk in chunks]
    facts = FinancialFactExtractor().extract(parsed, metrics=metrics or [])
    return {"facts": [fact.model_dump(mode="json") for fact in facts]}


class _CalcArgs(BaseModel):
    metric_name: str
    inputs: dict[str, str]
    periods: dict[str, str] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)


@tool("calculate_financial_metrics", args_schema=_CalcArgs)
def calculate_financial_metrics(
    metric_name: str,
    inputs: dict[str, str],
    periods: dict[str, str] | None = None,
    source_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate a supported financial metric using Decimal arithmetic."""
    parsed = {key: Decimal(value) for key, value in inputs.items()}
    result = CalculationEngine().calculate(
        metric_name,
        parsed,
        periods=periods,
        source_chunk_ids=source_chunk_ids,
    )
    return result.model_dump(mode="json")


class _ComparePeriodsArgs(BaseModel):
    metric_name: str
    current_value: str
    previous_value: str
    current_period: str | None = None
    previous_period: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)


@tool("compare_financial_periods", args_schema=_ComparePeriodsArgs)
def compare_financial_periods(
    metric_name: str,
    current_value: str,
    previous_value: str,
    current_period: str | None = None,
    previous_period: str | None = None,
    source_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare a metric across two periods via revenue_growth-style delta when applicable."""
    engine = CalculationEngine()
    if metric_name in {"revenue", "revenue_growth"}:
        result = engine.calculate(
            "revenue_growth",
            {
                "current_revenue": Decimal(current_value),
                "previous_revenue": Decimal(previous_value),
            },
            periods={
                "current": current_period or "",
                "previous": previous_period or "",
            },
            source_chunk_ids=source_chunk_ids,
        )
        return result.model_dump(mode="json")
    current = Decimal(current_value)
    previous = Decimal(previous_value)
    delta = current - previous
    return {
        "metric_name": metric_name,
        "current": str(current),
        "previous": str(previous),
        "delta": str(delta),
        "periods": {"current": current_period, "previous": previous_period},
        "source_chunk_ids": source_chunk_ids or [],
        "status": "success",
    }


class _CompareCompaniesArgs(BaseModel):
    metric_name: str
    values_by_ticker: dict[str, str]
    period: str | None = None


@tool("compare_companies", args_schema=_CompareCompaniesArgs)
def compare_companies(
    metric_name: str,
    values_by_ticker: dict[str, str],
    period: str | None = None,
) -> dict[str, Any]:
    """Compare a metric across company tickers using provided validated values."""
    parsed = {ticker: Decimal(value) for ticker, value in values_by_ticker.items()}
    ranking = sorted(parsed.items(), key=lambda item: item[1], reverse=True)
    return {
        "metric_name": metric_name,
        "period": period,
        "values": {k: str(v) for k, v in parsed.items()},
        "ranking": [{"ticker": t, "value": str(v)} for t, v in ranking],
        "status": "success",
    }


class _RiskArgs(BaseModel):
    ticker: str
    query: str = "Item 1A Risk Factors changes"


@tool("analyze_risk_changes", args_schema=_RiskArgs)
def analyze_risk_changes(ticker: str, query: str = "Item 1A Risk Factors changes") -> dict[str, Any]:
    """Retrieve risk-factor disclosures for qualitative comparison across years."""
    from ai.tools.search_tools import search_company_filings

    return search_company_filings.invoke(
        {
            "ticker": ticker,
            "query": query,
            "filing_types": ["10-K", "10-K/A"],
            "section": "Item 1A",
        }
    )


class _SummaryArgs(BaseModel):
    question: str
    evidence_snippets: list[str] = Field(default_factory=list)
    calculations: list[dict[str, Any]] = Field(default_factory=list)


@tool("build_financial_summary", args_schema=_SummaryArgs)
def build_financial_summary(
    question: str,
    evidence_snippets: list[str] | None = None,
    calculations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured summary skeleton from evidence snippets and calculations."""
    return {
        "question": question,
        "bullet_findings": (evidence_snippets or [])[:8],
        "calculations": calculations or [],
        "status": "success",
    }
