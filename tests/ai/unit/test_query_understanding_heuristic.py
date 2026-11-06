"""Heuristic query understanding without calling OpenRouter."""

from __future__ import annotations

from unittest.mock import patch

from ai.rag.query_understanding import QueryUnderstandingService
from ai.schemas.research import ResearchRequest


def test_heuristic_comparison_intent() -> None:
    service = QueryUnderstandingService()
    request = ResearchRequest(
        question="Compare NVIDIA and Microsoft revenue growth from 2021 to 2025.",
        selected_companies=[],
    )
    with patch.object(service, "_llm_intent", side_effect=RuntimeError("skip llm")):
        intent = service.understand(request)
    assert intent.task_type == "company_comparison"
    assert set(intent.tickers) == {"NVDA", "MSFT"}
    assert intent.start_year == 2021
    assert intent.end_year == 2025
    assert intent.requires_calculation is True
    assert intent.requires_multiple_namespaces is True
