"""Unit tests for RRF, compression, prioritization, and planner."""

from __future__ import annotations

from ai.agents.planner import ResearchPlanner
from ai.context.compressor import compress_chunks
from ai.context.prioritizer import prioritize_chunks
from ai.rag.hybrid_retriever import reciprocal_rank_fusion
from ai.schemas.research import ResearchIntent, ResearchRequest
from ai.schemas.retrieval import RetrievedChunk


def _chunk(chunk_id: str, ticker: str, score: float, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        ticker=ticker,
        text=text,
        score=score,
        form="10-K",
    )


def test_reciprocal_rank_fusion_prefers_consensus() -> None:
    list_a = [_chunk("a", "NVDA", 0.9), _chunk("b", "NVDA", 0.8), _chunk("c", "NVDA", 0.7)]
    list_b = [_chunk("b", "NVDA", 0.95), _chunk("a", "NVDA", 0.5), _chunk("d", "NVDA", 0.4)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].chunk_id in {"a", "b"}
    assert {c.chunk_id for c in fused} >= {"a", "b", "c", "d"}


def test_compress_dedupes_identical_text() -> None:
    chunks = [
        _chunk("1", "AAPL", 1.0, "Revenue increased."),
        _chunk("2", "AAPL", 0.9, "Revenue increased."),
        _chunk("3", "AAPL", 0.8, "Operating income declined."),
    ]
    compressed = compress_chunks(chunks)
    assert len(compressed) == 2


def test_prioritizer_prefers_matching_ticker() -> None:
    intent = ResearchIntent(
        task_type="company_overview",
        tickers=["MSFT"],
        search_queries=["Microsoft overview"],
    )
    chunks = [
        _chunk("1", "AAPL", 0.99),
        _chunk("2", "MSFT", 0.5),
    ]
    ordered = prioritize_chunks(chunks, intent)
    assert ordered[0].ticker == "MSFT"


def test_planner_includes_calculation_ops() -> None:
    request = ResearchRequest(
        question="Compare NVDA and MSFT revenue growth 2021-2025",
        selected_companies=["NVDA", "MSFT"],
    )
    intent = ResearchIntent(
        task_type="company_comparison",
        tickers=["NVDA", "MSFT"],
        metrics=["revenue_growth"],
        start_year=2021,
        end_year=2025,
        requires_calculation=True,
        requires_multiple_namespaces=True,
        search_queries=[request.question],
    )
    plan = ResearchPlanner().plan(request, intent)
    assert "retrieve_financial_evidence" in plan.operations
    assert "calculate_metrics" in plan.operations
    assert "compare_companies" in plan.operations
    assert plan.companies == ["NVDA", "MSFT"]
