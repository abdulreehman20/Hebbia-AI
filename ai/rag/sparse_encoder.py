"""BM25 sparse encoder helpers for optional hybrid retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class SparseEncoderService:
    """Fit/load a pinecone-text BM25Encoder for hybrid queries.

    Hybrid dense+sparse requires a Pinecone index with metric=dotproduct.
    Phase 2 indexes use cosine; this service is ready but gated by the retriever.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._encoder: Any | None = None

    @property
    def cache_path(self) -> Path:
        return Path(self.settings.bm25_cache_path)

    def load_or_none(self) -> Any | None:
        path = self.cache_path
        if not path.exists():
            return None
        from pinecone_text.sparse import BM25Encoder

        encoder = BM25Encoder()
        encoder.load(str(path))
        self._encoder = encoder
        return encoder

    def fit_and_save(self, corpus: list[str]) -> Any:
        if not corpus:
            raise ValueError("Cannot fit BM25 encoder on empty corpus")
        from pinecone_text.sparse import BM25Encoder

        encoder = BM25Encoder()
        encoder.fit(corpus)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        encoder.dump(str(self.cache_path))
        # Also write a tiny marker for tests / debugging.
        marker = self.cache_path.with_suffix(".meta.json")
        marker.write_text(
            json.dumps({"documents": len(corpus), "path": str(self.cache_path)}),
            encoding="utf-8",
        )
        self._encoder = encoder
        logger.info("Fitted BM25 encoder on %s documents -> %s", len(corpus), self.cache_path)
        return encoder

    def encode_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        encoder = self._encoder or self.load_or_none()
        if encoder is None:
            raise RuntimeError("BM25 encoder is not fitted; cannot encode sparse queries")
        return [encoder.encode_queries(query) for query in queries]

    def encode_documents(self, documents: list[str]) -> list[dict[str, Any]]:
        encoder = self._encoder or self.load_or_none()
        if encoder is None:
            raise RuntimeError("BM25 encoder is not fitted; cannot encode sparse documents")
        return [encoder.encode_documents(doc) for doc in documents]
