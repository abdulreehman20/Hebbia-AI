"""Build validated Pinecone metadata filters (never LLM-supplied dicts)."""

from __future__ import annotations

from typing import Any

from ai.schemas.retrieval import RetrievalFilters

ALLOWED_FORMS = frozenset(
    {
        "10-K",
        "10-Q",
        "8-K",
        "8-K/A",
        "10-K/A",
        "10-Q/A",
        "DEF 14A",
        "S-1",
        "S-8",
        "4",
    }
)


def build_pinecone_filter(filters: RetrievalFilters) -> dict[str, Any] | None:
    """Convert a RetrievalFilters schema into a Pinecone filter dict."""
    clauses: list[dict[str, Any]] = []

    if filters.forms:
        normalized: list[str] = []
        for form in filters.forms:
            candidate = form.strip()
            if not candidate:
                continue
            upper = candidate.upper()
            if upper not in ALLOWED_FORMS and not any(
                upper.startswith(prefix) for prefix in ("10-", "8-", "S-", "DEF")
            ):
                raise ValueError(f"Unsupported filing form in filter: {form}")
            normalized.append(candidate)
        if normalized:
            clauses.append({"form": {"$in": normalized}})

    if filters.filing_date_start or filters.filing_date_end:
        date_clause: dict[str, str] = {}
        if filters.filing_date_start:
            date_clause["$gte"] = filters.filing_date_start
        if filters.filing_date_end:
            date_clause["$lte"] = filters.filing_date_end
        clauses.append({"filing_date": date_clause})

    if filters.report_date_start or filters.report_date_end:
        report_clause: dict[str, str] = {}
        if filters.report_date_start:
            report_clause["$gte"] = filters.report_date_start
        if filters.report_date_end:
            report_clause["$lte"] = filters.report_date_end
        clauses.append({"report_date": report_clause})

    if filters.fiscal_year_start is not None or filters.fiscal_year_end is not None:
        year_clause: dict[str, int] = {}
        if filters.fiscal_year_start is not None:
            year_clause["$gte"] = int(filters.fiscal_year_start)
        if filters.fiscal_year_end is not None:
            year_clause["$lte"] = int(filters.fiscal_year_end)
        # Fiscal year may be derived from report_date year when not stored.
        # Prefer explicit fiscal_year metadata when present; also allow filing_date year match
        # via report_date range as a secondary filter when fiscal_year is absent in index.
        clauses.append({"fiscal_year": year_clause})

    if filters.sections:
        sections = [section.strip() for section in filters.sections if section.strip()]
        if sections:
            clauses.append({"section": {"$in": sections}})

    if filters.accession_numbers:
        accessions = [a.strip() for a in filters.accession_numbers if a.strip()]
        if accessions:
            clauses.append({"accession_number": {"$in": accessions}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def filters_from_years(
    *,
    forms: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    sections: list[str] | None = None,
) -> RetrievalFilters:
    """Convenience builder used by query understanding / tools."""
    return RetrievalFilters(
        forms=forms or [],
        fiscal_year_start=start_year,
        fiscal_year_end=end_year,
        sections=sections or [],
    )
