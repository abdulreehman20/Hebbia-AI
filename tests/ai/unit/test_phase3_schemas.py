"""Unit tests for Phase 3 schemas and namespace/filter helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai.calculations.formulas import calculate_metric
from ai.rag.citation_registry import CitationRegistry
from ai.rag.metadata_filters import build_pinecone_filter
from ai.rag.namespace_registry import get_namespace, validate_tickers
from ai.schemas.research import ResearchRequest
from ai.schemas.retrieval import RetrievedChunk, RetrievalFilters


def test_research_request_rejects_unknown_ticker() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(question="What is Tesla revenue?", selected_companies=["TSLA"])


def test_research_request_normalizes_tickers() -> None:
    req = ResearchRequest(question="Apple revenue", selected_companies=["aapl"])
    assert req.selected_companies == ["AAPL"]


def test_namespace_registry_allowlist() -> None:
    assert get_namespace("NVDA") == "nvidia"
    with pytest.raises(ValueError):
        get_namespace("TSLA")
    assert validate_tickers(["msft", "MSFT", "aapl"]) == ["MSFT", "AAPL"]


def test_metadata_filter_builder() -> None:
    filters = RetrievalFilters(forms=["10-K", "10-Q"], sections=["Item 7"])
    pinecone_filter = build_pinecone_filter(filters)
    assert pinecone_filter is not None
    assert "$and" in pinecone_filter or "form" in pinecone_filter


def test_revenue_growth_formula() -> None:
    result = calculate_metric(
        "revenue_growth",
        {
            "current_revenue": Decimal("120"),
            "previous_revenue": Decimal("100"),
        },
    )
    assert result.status == "success"
    assert result.result == Decimal("20")


def test_zero_denominator_is_invalid() -> None:
    result = calculate_metric(
        "gross_margin",
        {"gross_profit": Decimal("10"), "revenue": Decimal("0")},
    )
    assert result.status == "invalid_inputs"


def test_citation_registry_ids() -> None:
    registry = CitationRegistry()
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="Revenue was $100.",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form="10-K",
        filing_date="2025-02-26",
        section="Item 8",
    )
    citation = registry.register(chunk)
    assert citation.citation_id == "S1"
    assert "NVIDIA" in citation.label
    assert registry.validate_ids(["S1"]).valid is True
    assert registry.validate_ids(["S9"]).valid is False
