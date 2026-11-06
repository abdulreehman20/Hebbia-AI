"""Dense (and gated hybrid) Pinecone retrieval with RRF fusion."""

from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone

from ai.providers.openrouter_embeddings import OpenRouterEmbeddings
from ai.rag.metadata_filters import build_pinecone_filter
from ai.rag.namespace_registry import get_namespace, validate_tickers
from ai.rag.sparse_encoder import SparseEncoderService
from ai.schemas.retrieval import RetrievedChunk, RetrievalFilters, SearchResult
from config import Settings, get_settings

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge multiple ranked result lists with RRF."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            previous = best.get(chunk.chunk_id)
            if previous is None or (chunk.score or 0) > (previous.score or 0):
                best[chunk.chunk_id] = chunk
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id, rrf_score in ordered:
        chunk = best[chunk_id].model_copy(update={"score": rrf_score})
        fused.append(chunk)
    return fused


class HybridRetriever:
    """Retrieve from company namespaces using dense vectors; hybrid when allowed."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: OpenRouterEmbeddings | None = None,
        sparse: SparseEncoderService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or OpenRouterEmbeddings(self.settings)
        self.sparse = sparse or SparseEncoderService(self.settings)
        self._index: Any | None = None
        self._metric: str | None = None
        self._hybrid_enabled: bool | None = None

    def _client_index(self) -> Any:
        if self._index is not None:
            return self._index
        self.settings.require_pinecone()
        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        name = self.settings.pinecone_index_name
        description = pc.describe_index(name)
        metric = getattr(description, "metric", None)
        if metric is None and isinstance(description, dict):
            metric = description.get("metric")
        self._metric = str(metric or "unknown").lower()
        self._hybrid_enabled = self._metric == "dotproduct"
        if not self._hybrid_enabled:
            logger.info(
                "Pinecone index metric=%s — hybrid dense+sparse disabled; using dense+RRF",
                self._metric,
            )
        self._index = pc.Index(name)
        return self._index

    @property
    def hybrid_enabled(self) -> bool:
        if self._hybrid_enabled is None:
            self._client_index()
        return bool(self._hybrid_enabled)

    @property
    def index_metric(self) -> str | None:
        if self._metric is None:
            try:
                self._client_index()
            except Exception:  # noqa: BLE001
                return None
        return self._metric

    def retrieve(
        self,
        *,
        ticker: str,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> SearchResult:
        tickers = validate_tickers([ticker])
        ticker = tickers[0]
        namespace = get_namespace(ticker)
        limit = top_k or self.settings.retrieval_top_k
        pinecone_filter = build_pinecone_filter(filters or RetrievalFilters())
        dense = self.embeddings.embed_query(query)

        index = self._client_index()
        kwargs: dict[str, Any] = {
            "vector": dense,
            "top_k": limit,
            "namespace": namespace,
            "include_metadata": True,
        }
        if pinecone_filter:
            kwargs["filter"] = pinecone_filter

        if self.hybrid_enabled:
            try:
                sparse_vec = self.sparse.encode_queries([query])[0]
                kwargs["sparse_vector"] = sparse_vec
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sparse encode failed; falling back to dense-only: %s", exc)

        response = index.query(**kwargs)
        matches = getattr(response, "matches", None) or response.get("matches", [])
        chunks = [self._match_to_chunk(match, ticker=ticker, namespace=namespace) for match in matches]
        return SearchResult(query=query, ticker=ticker, namespace=namespace, chunks=chunks)

    def retrieve_many(
        self,
        *,
        tickers: list[str],
        queries: list[str],
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run dense retrieval across tickers/queries and fuse with RRF."""
        validated = validate_tickers(tickers)
        ranked_lists: list[list[RetrievedChunk]] = []
        calls = 0
        max_calls = self.settings.max_retrieval_calls
        for ticker in validated:
            for query in queries:
                if calls >= max_calls:
                    logger.warning("Hit max_retrieval_calls=%s; stopping early", max_calls)
                    break
                result = self.retrieve(
                    ticker=ticker,
                    query=query,
                    filters=filters,
                    top_k=top_k,
                )
                ranked_lists.append(result.chunks)
                calls += 1
            if calls >= max_calls:
                break
        return reciprocal_rank_fusion(ranked_lists)

    @staticmethod
    def _match_to_chunk(match: Any, *, ticker: str, namespace: str) -> RetrievedChunk:
        if isinstance(match, dict):
            chunk_id = str(match.get("id") or "")
            score = match.get("score")
            metadata = match.get("metadata") or {}
        else:
            chunk_id = str(getattr(match, "id", "") or "")
            score = getattr(match, "score", None)
            metadata = getattr(match, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return RetrievedChunk(
            chunk_id=chunk_id,
            score=float(score) if score is not None else None,
            text=str(metadata.get("text") or ""),
            ticker=str(metadata.get("ticker") or ticker),
            company_name=metadata.get("company_name"),
            form=metadata.get("form"),
            filing_date=metadata.get("filing_date"),
            report_date=metadata.get("report_date"),
            accession_number=metadata.get("accession_number"),
            source_url=metadata.get("source_url"),
            source_document=metadata.get("source_document"),
            section=metadata.get("section"),
            chunk_index=metadata.get("chunk_index"),
            namespace=namespace,
        )
