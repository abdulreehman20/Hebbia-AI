"""Research request and intent schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from constants import SUPPORTED_TICKERS

TaskType = Literal[
    "company_overview",
    "financial_statement_analysis",
    "revenue_and_profitability",
    "historical_trends",
    "business_segment_analysis",
    "risk_analysis",
    "management_discussion",
    "corporate_events",
    "cash_flow_and_capital",
    "accounting_and_footnotes",
    "financial_calculations",
    "company_comparison",
    "investment_due_diligence",
    "unsupported",
]


class ResearchRequest(BaseModel):
    """Inbound research question from an investor."""

    question: str = Field(min_length=3)
    selected_companies: list[str] = Field(default_factory=list)
    reporting_period_start: str | None = None
    reporting_period_end: str | None = None
    filing_types: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    response_preferences: dict[str, str] = Field(default_factory=dict)

    @field_validator("selected_companies")
    @classmethod
    def normalize_tickers(cls, values: list[str]) -> list[str]:
        normalized = [ticker.strip().upper() for ticker in values if ticker.strip()]
        unknown = [ticker for ticker in normalized if ticker not in SUPPORTED_TICKERS]
        if unknown:
            raise ValueError(f"Unsupported tickers: {unknown}")
        return normalized


class ResearchIntent(BaseModel):
    """Structured understanding of an investor question."""

    task_type: TaskType
    tickers: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None
    filing_types: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    requires_calculation: bool = False
    requires_multiple_namespaces: bool = False
    search_queries: list[str] = Field(default_factory=list)
    unsupported_reason: str | None = None

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, values: list[str]) -> list[str]:
        normalized = [ticker.strip().upper() for ticker in values if ticker.strip()]
        unknown = [ticker for ticker in normalized if ticker not in SUPPORTED_TICKERS]
        if unknown:
            raise ValueError(f"Unsupported tickers: {unknown}")
        return normalized
