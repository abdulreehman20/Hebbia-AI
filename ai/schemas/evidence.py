"""Evidence package schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.schemas.citations import SourceCitation
from ai.schemas.financial import CalculationResult, FinancialFact
from ai.schemas.plan import ResearchPlan
from ai.schemas.retrieval import RetrievedChunk


class EvidenceSource(BaseModel):
    """Source provenance for an evidence item."""

    chunk_id: str
    ticker: str
    form: str | None = None
    filing_date: str | None = None
    accession_number: str | None = None
    source_url: str | None = None
    section: str | None = None


class EvidenceItem(BaseModel):
    """One piece of retrieved evidence with citation id."""

    citation_id: str
    chunk: RetrievedChunk
    source: EvidenceSource


class MissingEvidence(BaseModel):
    """Record of unavailable requested evidence."""

    description: str
    tickers: list[str] = Field(default_factory=list)
    metric: str | None = None
    period: str | None = None


class EvidencePackage(BaseModel):
    """Bundled evidence for answer generation."""

    question: str
    objective: str | None = None
    companies: list[str] = Field(default_factory=list)
    reporting_periods: list[str] = Field(default_factory=list)
    plan: ResearchPlan | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    financial_facts: list[FinancialFact] = Field(default_factory=list)
    calculations: list[CalculationResult] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    missing: list[MissingEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
