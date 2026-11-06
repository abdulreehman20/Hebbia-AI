"""Citation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Human-readable citation for an evidence chunk."""

    citation_id: str
    label: str
    chunk_id: str
    ticker: str
    form: str | None = None
    filing_date: str | None = None
    section: str | None = None
    source_url: str | None = None


class CitationReference(BaseModel):
    """Inline citation reference used in generated text."""

    citation_id: str
    claim: str | None = None


class CitationValidationResult(BaseModel):
    """Outcome of validating citations against evidence."""

    valid: bool
    invalid_citation_ids: list[str] = Field(default_factory=list)
    missing_for_claims: list[str] = Field(default_factory=list)
    message: str | None = None
