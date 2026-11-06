"""High-level retrieval orchestration: understand → rewrite → retrieve → rerank."""

from __future__ import annotations

import logging

from ai.rag.hybrid_retriever import HybridRetriever
from ai.rag.metadata_filters import filters_from_years
from ai.rag.query_rewriter import QueryRewriter
from ai.rag.query_understanding import QueryUnderstandingService
from ai.rag.reranking import RerankingService
from ai.schemas.research import ResearchIntent, ResearchRequest
from ai.schemas.retrieval import RerankedResult, RetrievedChunk, RetrievalFilters
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """End-to-end retrieval for a research request."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        understanding: QueryUnderstandingService | None = None,
        rewriter: QueryRewriter | None = None,
        retriever: HybridRetriever | None = None,
        reranker: RerankingService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.understanding = understanding or QueryUnderstandingService(self.settings)
        self.rewriter = rewriter or QueryRewriter(self.settings)
        self.retriever = retriever or HybridRetriever(self.settings)
        self.reranker = reranker or RerankingService(self.settings)

    def understand(self, request: ResearchRequest) -> ResearchIntent:
        return self.understanding.understand(request)

    def retrieve_for_intent(
        self,
        question: str,
        intent: ResearchIntent,
        *,
        filters: RetrievalFilters | None = None,
    ) -> RerankedResult:
        if not intent.tickers:
            return RerankedResult(question=question, chunks=[])
        queries = self.rewriter.rewrite(question, intent)
        effective_filters = filters or filters_from_years(
            forms=intent.filing_types or None,
            start_year=intent.start_year,
            end_year=intent.end_year,
            sections=intent.sections or None,
        )
        # Drop fiscal_year filter when index may not have fiscal_year metadata
        # (Phase 2 stores filing_date/report_date). Prefer date-free + post filter.
        safe_filters = RetrievalFilters(
            forms=effective_filters.forms,
            sections=effective_filters.sections,
            accession_numbers=effective_filters.accession_numbers,
            filing_date_start=effective_filters.filing_date_start,
            filing_date_end=effective_filters.filing_date_end,
            report_date_start=effective_filters.report_date_start,
            report_date_end=effective_filters.report_date_end,
            # omit fiscal_year_* until metadata is confirmed populated
        )
        candidates = self.retriever.retrieve_many(
            tickers=intent.tickers,
            queries=queries,
            filters=safe_filters,
        )
        if intent.start_year or intent.end_year:
            candidates = self._filter_by_year(candidates, intent.start_year, intent.end_year)
        reranked = self.reranker.rerank(question, candidates, tickers=intent.tickers)
        return RerankedResult(question=question, chunks=reranked)

    @staticmethod
    def _filter_by_year(
        chunks: list[RetrievedChunk],
        start_year: int | None,
        end_year: int | None,
    ) -> list[RetrievedChunk]:
        if start_year is None and end_year is None:
            return chunks
        kept: list[RetrievedChunk] = []
        for chunk in chunks:
            year = None
            for date_value in (chunk.report_date, chunk.filing_date):
                if date_value and len(date_value) >= 4 and date_value[:4].isdigit():
                    year = int(date_value[:4])
                    break
            if year is None:
                kept.append(chunk)
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            kept.append(chunk)
        return kept or chunks
