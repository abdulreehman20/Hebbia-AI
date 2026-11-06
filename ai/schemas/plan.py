"""Research planning schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ai.schemas.research import TaskType

OperationType = Literal[
    "retrieve_financial_evidence",
    "extract_financial_facts",
    "calculate_metrics",
    "compare_companies",
    "compare_periods",
    "analyze_risks",
    "verify_citations",
    "check_missing_information",
]


class ResearchTask(BaseModel):
    """A single planned research operation."""

    operation: OperationType
    tickers: list[str] = Field(default_factory=list)
    query: str | None = None
    metrics: list[str] = Field(default_factory=list)
    filing_types: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Validated research plan for the single Financial Research Agent."""

    task_type: TaskType
    objective: str
    companies: list[str] = Field(default_factory=list)
    reporting_period_start: int | None = None
    reporting_period_end: int | None = None
    filing_types: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    requires_calculation: bool = False
    requires_comparison: bool = False
    operations: list[OperationType] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
