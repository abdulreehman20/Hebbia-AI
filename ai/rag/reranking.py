"""Reranking facade with multi-company coverage guard."""

from __future__ import annotations

from collections import defaultdict

from ai.providers.local_reranker import LocalReranker
from ai.schemas.retrieval import RetrievedChunk
from config import Settings, get_settings


class RerankingService:
    """Rerank candidates and preserve per-ticker coverage when possible."""

    def __init__(
        self,
        settings: Settings | None = None,
        reranker: LocalReranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.reranker = reranker or LocalReranker(self.settings)

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        tickers: list[str] | None = None,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        limit = top_n or self.settings.rerank_top_n
        scored = self.reranker.rerank(question, chunks, top_n=max(limit, len(chunks)))
        if not tickers or len(tickers) <= 1:
            return scored[:limit]

        # Ensure each requested ticker keeps at least one chunk when available.
        by_ticker: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in scored:
            by_ticker[chunk.ticker.upper()].append(chunk)

        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        for ticker in tickers:
            pool = by_ticker.get(ticker.upper(), [])
            if pool and pool[0].chunk_id not in selected_ids:
                selected.append(pool[0])
                selected_ids.add(pool[0].chunk_id)

        for chunk in scored:
            if len(selected) >= limit:
                break
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
        return selected[:limit]
