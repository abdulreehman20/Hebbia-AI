"""Document search tools backed by Pinecone namespaces."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.metadata_filters import filters_from_years
from ai.rag.namespace_registry import validate_tickers
from ai.schemas.retrieval import RetrievalFilters


class _SearchArgs(BaseModel):
    ticker: str
    query: str
    filing_types: list[str] = Field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None
    section: str | None = None


def _retriever() -> HybridRetriever:
    return HybridRetriever()


def _serialize_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    return [chunk.model_dump() for chunk in chunks]


@tool("search_company_filings", args_schema=_SearchArgs)
def search_company_filings(
    ticker: str,
    query: str,
    filing_types: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    """Search one company's Pinecone namespace for SEC filing evidence."""
    ticker = validate_tickers([ticker])[0]
    filters = filters_from_years(
        forms=filing_types or None,
        start_year=start_year,
        end_year=end_year,
        sections=[section] if section else None,
    )
    # Avoid fiscal_year metadata dependency.
    filters = RetrievalFilters(forms=filters.forms, sections=filters.sections)
    result = _retriever().retrieve(ticker=ticker, query=query, filters=filters)
    return {
        "ticker": ticker,
        "namespace": result.namespace,
        "query": query,
        "chunks": _serialize_chunks(result.chunks[:15]),
    }


class _MultiSearchArgs(BaseModel):
    tickers: list[str]
    query: str
    filing_types: list[str] = Field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None


@tool("search_multiple_companies", args_schema=_MultiSearchArgs)
def search_multiple_companies(
    tickers: list[str],
    query: str,
    filing_types: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, Any]:
    """Search multiple approved company namespaces for comparative research."""
    tickers = validate_tickers(tickers)
    filters = RetrievalFilters(forms=filing_types or [])
    chunks = _retriever().retrieve_many(
        tickers=tickers,
        queries=[query],
        filters=filters,
    )
    return {"tickers": tickers, "query": query, "chunks": _serialize_chunks(chunks[:20])}


class _FilterArgs(BaseModel):
    filing_types: list[str] = Field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None
    sections: list[str] = Field(default_factory=list)


@tool("filter_filings", args_schema=_FilterArgs)
def filter_filings(
    filing_types: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Build a validated metadata filter object for subsequent retrieval."""
    filters = filters_from_years(
        forms=filing_types,
        start_year=start_year,
        end_year=end_year,
        sections=sections,
    )
    return filters.model_dump()


class _SectionArgs(BaseModel):
    ticker: str
    query: str
    section: str
    filing_types: list[str] = Field(default_factory=list)


@tool("search_filing_section", args_schema=_SectionArgs)
def search_filing_section(
    ticker: str,
    query: str,
    section: str,
    filing_types: list[str] | None = None,
) -> dict[str, Any]:
    """Search a specific SEC section such as Risk Factors or MD&A."""
    return search_company_filings.invoke(
        {
            "ticker": ticker,
            "query": query,
            "filing_types": filing_types or [],
            "section": section,
        }
    )
