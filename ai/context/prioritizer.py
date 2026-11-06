"""Prioritize evidence chunks for the context window."""

from __future__ import annotations

from ai.schemas.research import ResearchIntent
from ai.schemas.retrieval import RetrievedChunk


def prioritize_chunks(
    chunks: list[RetrievedChunk],
    intent: ResearchIntent | None = None,
) -> list[RetrievedChunk]:
    """Sort chunks by relevance heuristics (company, period, form, score)."""

    def score(chunk: RetrievedChunk) -> tuple:
        ticker_match = 0
        period_match = 0
        form_match = 0
        section_bonus = 0
        if intent:
            if chunk.ticker.upper() in {t.upper() for t in intent.tickers}:
                ticker_match = 1
            year = None
            for date_value in (chunk.report_date, chunk.filing_date):
                if date_value and len(date_value) >= 4 and date_value[:4].isdigit():
                    year = int(date_value[:4])
                    break
            if year is not None:
                if intent.start_year and intent.end_year:
                    period_match = 1 if intent.start_year <= year <= intent.end_year else 0
                elif intent.start_year:
                    period_match = 1 if year == intent.start_year else 0
            if intent.filing_types and chunk.form in intent.filing_types:
                form_match = 1
            if intent.sections and chunk.section in intent.sections:
                section_bonus = 1
        numeric_score = chunk.score if chunk.score is not None else 0.0
        return (ticker_match, period_match, form_match, section_bonus, numeric_score)

    return sorted(chunks, key=score, reverse=True)
