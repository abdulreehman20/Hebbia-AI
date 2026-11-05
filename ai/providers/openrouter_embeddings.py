"""OpenRouter embedding client compatible with LangChain Embeddings."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    """Raised when OpenRouter keeps returning HTTP 429 after retries."""


class OpenRouterEmbeddings(Embeddings):
    """Generate embeddings through the OpenRouter embeddings endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.require_openrouter()
        self.model = self.settings.openrouter_embedding_model
        self.url = self.settings.openrouter_embeddings_url
        self.timeout = self.settings.request_timeout_seconds
        self.max_retries = self.settings.max_retries
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hebbia/backend",
            "X-Title": "Financial Research Agentic RAG",
        }

    @staticmethod
    def _retry_after_seconds(response: httpx.Response | None) -> float | None:
        if response is None or response.headers is None:
            return None
        raw = response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _backoff_seconds(
        self,
        attempt: int,
        *,
        status_code: int | None,
        response: httpx.Response | None,
    ) -> float:
        """Compute free-tier-friendly backoff; honor Retry-After on 429."""
        retry_after = self._retry_after_seconds(response)
        exponential = float(2**attempt)
        if status_code == 429:
            # Free models often need longer cool-downs than 1–16s.
            base = max(retry_after or 0.0, exponential * 2.0, 15.0)
            return min(base, self.settings.rate_limit_max_backoff_seconds)
        if retry_after is not None:
            return min(max(retry_after, exponential), self.settings.rate_limit_max_backoff_seconds)
        return min(exponential, 60.0)

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }

        last_error: Exception | None = None
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.url, headers=self._headers(), json=payload)
                last_status = response.status_code
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                data = body.get("data")
                if not isinstance(data, list) or not data:
                    raise ValueError(f"Malformed embeddings response: {body!r}")

                # Preserve input order when OpenRouter returns indexed rows.
                ordered = sorted(
                    data,
                    key=lambda item: int(item.get("index", 0))
                    if isinstance(item, dict)
                    else 0,
                )
                vectors: list[list[float]] = []
                for item in ordered:
                    if not isinstance(item, dict) or "embedding" not in item:
                        raise ValueError(f"Missing embedding in response item: {item!r}")
                    vector = item["embedding"]
                    if not isinstance(vector, list) or not vector:
                        raise ValueError("Empty embedding vector returned")
                    vectors.append([float(value) for value in vector])

                if len(vectors) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, received {len(vectors)}"
                    )

                dims = {len(vector) for vector in vectors}
                if len(dims) != 1:
                    raise ValueError(f"Inconsistent embedding dimensions: {dims}")
                dimension = next(iter(dims))
                if self._dimension is None:
                    self._dimension = dimension
                elif self._dimension != dimension:
                    raise ValueError(
                        f"Embedding dimension changed from {self._dimension} to {dimension}"
                    )
                return vectors
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                status_code = last_status
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    status_code = exc.response.status_code
                    response = exc.response
                if attempt >= self.max_retries:
                    break
                delay = self._backoff_seconds(
                    attempt,
                    status_code=status_code,
                    response=response,
                )
                logger.warning(
                    "Embedding request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        if last_status == 429:
            raise RateLimitError(
                f"OpenRouter rate-limited (429) after retries: {last_error}"
            )
        raise RuntimeError(f"OpenRouter embeddings failed after retries: {last_error}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch in a single OpenRouter request."""
        return self._request_embeddings(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, self.settings.embedding_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.embed_batch(batch))
            # Pace free-tier calls when embed_documents receives a large list.
            if start + batch_size < len(texts) and self.settings.embedding_batch_pause_seconds > 0:
                time.sleep(self.settings.embedding_batch_pause_seconds)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0]

    def probe_dimension(self) -> int:
        """Embed a tiny probe string and return the vector dimension."""
        if self._dimension is not None:
            return self._dimension
        self.embed_query("dimension probe")
        if self._dimension is None:
            raise RuntimeError("Failed to determine embedding dimension")
        return self._dimension
