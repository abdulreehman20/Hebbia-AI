"""Retrieval-related schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    """Validated filters used to build Pinecone metadata queries."""

    forms: list[str] = Field(default_factory=list)
    filing_date_start: str | None = None
    filing_date_end: str | None = None
    report_date_start: str | None = None
    report_date_end: str | None = None
    fiscal_year_start: int | None = None
    fiscal_year_end: int | None = None
    sections: list[str] = Field(default_factory=list)
    accession_numbers: list[str] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    """A single retrieval request against one namespace."""

    ticker: str
    query: str
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    top_k: int = 30


class RetrievedChunk(BaseModel):
    """One retrieved evidence chunk."""

    chunk_id: str
    score: float | None = None
    text: str
    ticker: str
    company_name: str | None = None
    form: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    accession_number: str | None = None
    source_url: str | None = None
    source_document: str | None = None
    section: str | None = None
    chunk_index: int | None = None
    namespace: str | None = None


class SearchResult(BaseModel):
    """Retrieval results for one query/namespace."""

    query: str
    ticker: str
    namespace: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class RerankedResult(BaseModel):
    """Reranked evidence set."""

    question: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
