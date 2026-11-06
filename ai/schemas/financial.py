"""Financial fact and calculation schemas."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class FinancialPeriod(BaseModel):
    """Reporting period for a financial figure."""

    fiscal_year: int | None = None
    fiscal_quarter: str | None = None
    label: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class FinancialMetric(BaseModel):
    """Named metric definition."""

    name: str
    unit: str | None = None


class FinancialFact(BaseModel):
    """A numerical fact extracted from retrieved evidence."""

    metric_name: str
    value: Decimal
    currency: str | None = "USD"
    unit: str | None = None
    scale: str | None = None
    period: FinancialPeriod | None = None
    fiscal_year: int | None = None
    ticker: str
    accession_number: str | None = None
    chunk_id: str
    quotation: str | None = None


class FinancialCalculation(BaseModel):
    """Requested deterministic calculation."""

    metric_name: str
    formula: str
    inputs: dict[str, Decimal] = Field(default_factory=dict)
    periods: dict[str, str] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)


CalculationStatus = Literal["success", "missing_inputs", "invalid_inputs", "error"]


class CalculationResult(BaseModel):
    """Result of a deterministic financial calculation."""

    metric_name: str
    formula: str
    inputs: dict[str, Decimal] = Field(default_factory=dict)
    periods: dict[str, str] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)
    result: Decimal | None = None
    unit: str | None = None
    status: CalculationStatus
    message: str | None = None
