"""Final research answer schema."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.schemas.citations import SourceCitation
from ai.schemas.financial import CalculationResult


class ResearchAnswer(BaseModel):
    """Structured financial research response."""

    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    financial_analysis: str | None = None
    calculations: list[CalculationResult] = Field(default_factory=list)
    company_comparisons: str | None = None
    risks_and_limitations: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    raw_markdown: str | None = None
