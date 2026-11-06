"""Local BGE cross-encoder reranker."""

from __future__ import annotations

import logging
from typing import Any

from config import Settings, get_settings
from ai.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class LocalReranker:
    """Score query-document pairs with a local cross-encoder."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.reranker_model
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        # Lazy import: sentence-transformers/torch are heavy.
        from sentence_transformers import CrossEncoder

        logger.info("Loading local reranker model %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_n chunks sorted by cross-encoder relevance."""
        if not chunks:
            return []
        limit = top_n if top_n is not None else self.settings.rerank_top_n
        if len(chunks) <= limit:
            # Still score so callers get consistent ordering when model loads.
            pass

        try:
            model = self._load()
            pairs = [(question, chunk.text) for chunk in chunks]
            scores = [float(score) for score in model.predict(pairs)]
        except Exception as exc:  # noqa: BLE001 — keep retrieval usable without GPU/model
            logger.warning("Local reranker unavailable; falling back to retrieval scores: %s", exc)
            scored_fallback = sorted(
                chunks,
                key=lambda item: float(item.score or 0.0),
                reverse=True,
            )
            return scored_fallback[: max(1, limit)]

        scored = list(zip(chunks, scores, strict=True))
        scored.sort(key=lambda item: float(item[1]), reverse=True)

        result: list[RetrievedChunk] = []
        for chunk, score in scored[: max(1, limit)]:
            updated = chunk.model_copy(update={"score": float(score)})
            result.append(updated)
        return result
