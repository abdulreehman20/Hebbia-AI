"""Expand evidence with neighboring chunks from the same filing."""

from __future__ import annotations

import logging
from typing import Any

from pinecone import Pinecone

from ai.rag.namespace_registry import get_namespace
from ai.schemas.retrieval import RetrievedChunk
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class ContextExpansionService:
    """Fetch neighboring chunks for a selected evidence item when needed."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index: Any | None = None

    def _index_client(self) -> Any:
        if self._index is not None:
            return self._index
        self.settings.require_pinecone()
        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self._index = pc.Index(self.settings.pinecone_index_name)
        return self._index

    def expand(
        self,
        chunk: RetrievedChunk,
        *,
        window: int = 1,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the chunk plus nearby chunks from the same accession/document.

        Pinecone serverless does not support arbitrary ID prefix scans cheaply.
        Strategy: metadata-filtered dense query using the chunk's own text embedding
        (or provided vector), then keep same accession_number neighbors by chunk_index.
        """
        if window <= 0:
            return [chunk]
        if chunk.chunk_index is None or not chunk.accession_number:
            return [chunk]

        namespace = chunk.namespace or get_namespace(chunk.ticker)
        index = self._index_client()
        vector = query_vector
        if vector is None:
            # Use a zero-ish probe via metadata-only isn't supported; require caller embedding
            # or skip expansion.
            logger.info("No query_vector for expansion; returning original chunk only")
            return [chunk]

        response = index.query(
            vector=vector,
            top_k=min(20, 5 + window * 4),
            namespace=namespace,
            include_metadata=True,
            filter={
                "accession_number": {"$eq": chunk.accession_number},
            },
        )
        matches = getattr(response, "matches", None) or response.get("matches", [])
        neighbors: list[RetrievedChunk] = []
        target_index = int(chunk.chunk_index)
        for match in matches:
            meta = getattr(match, "metadata", None) if not isinstance(match, dict) else match.get("metadata")
            meta = meta or {}
            idx = meta.get("chunk_index")
            if idx is None:
                continue
            if abs(int(idx) - target_index) <= window:
                neighbors.append(
                    RetrievedChunk(
                        chunk_id=str(getattr(match, "id", None) or match.get("id") or ""),
                        score=float(getattr(match, "score", None) or match.get("score") or 0),
                        text=str(meta.get("text") or ""),
                        ticker=str(meta.get("ticker") or chunk.ticker),
                        company_name=meta.get("company_name"),
                        form=meta.get("form"),
                        filing_date=meta.get("filing_date"),
                        report_date=meta.get("report_date"),
                        accession_number=meta.get("accession_number"),
                        source_url=meta.get("source_url"),
                        source_document=meta.get("source_document"),
                        section=meta.get("section"),
                        chunk_index=meta.get("chunk_index"),
                        namespace=namespace,
                    )
                )
        if not neighbors:
            return [chunk]
        neighbors.sort(key=lambda item: item.chunk_index or 0)
        # Ensure original is present.
        ids = {item.chunk_id for item in neighbors}
        if chunk.chunk_id not in ids:
            neighbors.append(chunk)
            neighbors.sort(key=lambda item: item.chunk_index or 0)
        return neighbors
