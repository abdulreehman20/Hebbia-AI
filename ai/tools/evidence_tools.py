"""Evidence verification tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai.rag.citation_registry import CitationRegistry
from ai.schemas.citations import SourceCitation
from ai.schemas.financial import CalculationResult


class _VerifyEvidenceArgs(BaseModel):
    question: str
    chunk_texts: list[str] = Field(default_factory=list)


@tool("verify_retrieved_evidence", args_schema=_VerifyEvidenceArgs)
def verify_retrieved_evidence(question: str, chunk_texts: list[str] | None = None) -> dict[str, Any]:
    """Check that retrieved evidence is non-empty for the question."""
    texts = chunk_texts or []
    return {
        "question": question,
        "evidence_count": len(texts),
        "sufficient": len(texts) > 0,
        "message": None if texts else "No evidence chunks available",
    }


class _VerifyCalcArgs(BaseModel):
    calculations: list[dict[str, Any]] = Field(default_factory=list)


@tool("verify_financial_calculations", args_schema=_VerifyCalcArgs)
def verify_financial_calculations(calculations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Validate calculation status fields."""
    results = [CalculationResult.model_validate(item) for item in (calculations or [])]
    invalid = [r.metric_name for r in results if r.status != "success"]
    return {
        "valid": not invalid,
        "invalid_metrics": invalid,
        "count": len(results),
    }


class _VerifyCitationsArgs(BaseModel):
    citations: list[dict[str, Any]] = Field(default_factory=list)
    referenced_ids: list[str] = Field(default_factory=list)


@tool("verify_citations", args_schema=_VerifyCitationsArgs)
def verify_citations(
    citations: list[dict[str, Any]] | None = None,
    referenced_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Check that referenced citation ids exist in the registry snapshot."""
    registry = CitationRegistry()
    for item in citations or []:
        citation = SourceCitation.model_validate(item)
        # Reconstruct registry membership without chunks.
        registry._citations.append(citation)  # noqa: SLF001
        registry._by_id[citation.citation_id] = citation  # noqa: SLF001
        registry._by_chunk[citation.chunk_id] = citation  # noqa: SLF001
    return registry.validate_ids(referenced_ids or []).model_dump()


class _MissingArgs(BaseModel):
    requested_metrics: list[str] = Field(default_factory=list)
    available_metrics: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)


@tool("check_missing_information", args_schema=_MissingArgs)
def check_missing_information(
    requested_metrics: list[str] | None = None,
    available_metrics: list[str] | None = None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Identify requested metrics that were not found in evidence."""
    requested = requested_metrics or []
    available = set(available_metrics or [])
    missing = [metric for metric in requested if metric not in available]
    return {
        "missing_metrics": missing,
        "tickers": tickers or [],
        "complete": not missing,
    }
