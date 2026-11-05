"""Pinecone index management and dense vector upserts."""

from __future__ import annotations

import logging
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import Settings, get_settings
from constants import COMPANY_NAMESPACES, INGESTION_VERSION

logger = logging.getLogger(__name__)

# Pinecone metadata string values should stay well under practical limits.
MAX_TEXT_METADATA_CHARS = 35_000


def get_namespace(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in COMPANY_NAMESPACES:
        raise ValueError(f"Unsupported ticker for namespace mapping: {ticker}")
    return COMPANY_NAMESPACES[ticker]


def _client(settings: Settings) -> Pinecone:
    settings.require_pinecone()
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(dimension: int, settings: Settings | None = None) -> Any:
    """Inspect or create the financial-research dense index."""
    settings = settings or get_settings()
    pc = _client(settings)
    name = settings.pinecone_index_name
    existing = {index["name"] for index in pc.list_indexes()}

    if name in existing:
        description = pc.describe_index(name)
        # Support both attribute and dict-style SDK responses.
        existing_dim = getattr(description, "dimension", None)
        if existing_dim is None and isinstance(description, dict):
            existing_dim = description.get("dimension")
        existing_metric = getattr(description, "metric", None)
        if existing_metric is None and isinstance(description, dict):
            existing_metric = description.get("metric")

        if existing_dim != dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {existing_dim}, "
                f"but embeddings are {dimension}. Refusing to recreate or wipe."
            )
        logger.info(
            "Using existing Pinecone index '%s' (dimension=%s metric=%s)",
            name,
            existing_dim,
            existing_metric,
        )
        return pc.Index(name)

    logger.info(
        "Creating Pinecone index '%s' (dimension=%s metric=cosine cloud=%s region=%s)",
        name,
        dimension,
        settings.pinecone_cloud,
        settings.pinecone_region,
    )
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Wait briefly for readiness.
    for _ in range(30):
        description = pc.describe_index(name)
        status = getattr(description, "status", None)
        ready = False
        if status is not None:
            ready = getattr(status, "ready", False) or (
                isinstance(status, dict) and status.get("ready")
            )
        if ready:
            break
        time.sleep(2)
    return pc.Index(name)


def _metadata_for_chunk(chunk_meta: dict[str, Any], text: str) -> dict[str, Any]:
    truncated = text if len(text) <= MAX_TEXT_METADATA_CHARS else text[:MAX_TEXT_METADATA_CHARS]
    metadata = {
        "ticker": str(chunk_meta.get("ticker") or ""),
        "company_name": str(chunk_meta.get("company_name") or ""),
        "cik": str(chunk_meta.get("cik") or ""),
        "form": str(chunk_meta.get("form") or ""),
        "filing_date": str(chunk_meta.get("filing_date") or ""),
        "report_date": str(chunk_meta.get("report_date") or ""),
        "accession_number": str(chunk_meta.get("accession_number") or ""),
        "source_url": str(chunk_meta.get("source_url") or ""),
        "source_document": str(chunk_meta.get("source_document") or ""),
        "section": str(chunk_meta.get("section") or ""),
        "chunk_index": int(chunk_meta.get("chunk_index") or 0),
        "text": truncated,
        "embedding_model": str(chunk_meta.get("embedding_model") or ""),
        "ingestion_version": str(chunk_meta.get("ingestion_version") or INGESTION_VERSION),
    }
    return metadata


def upsert_chunks(
    index: Any,
    chunk_records: list[dict[str, Any]],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Upsert dense vectors grouped by company namespace.

    Each record: {id, values, metadata_source, text, ticker}
    """
    settings = settings or get_settings()
    batch_size = max(1, settings.pinecone_upsert_batch_size)
    success = 0
    failed: list[dict[str, Any]] = []
    by_namespace: dict[str, int] = {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in chunk_records:
        ticker = str(record["ticker"]).upper()
        namespace = get_namespace(ticker)
        grouped.setdefault(namespace, []).append(record)

    for namespace, records in grouped.items():
        logger.info("[%s] Uploading embeddings to namespace...", namespace)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors = []
            for item in batch:
                vectors.append(
                    {
                        "id": item["id"],
                        "values": item["values"],
                        "metadata": _metadata_for_chunk(item["metadata"], item["text"]),
                    }
                )
            try:
                index.upsert(vectors=vectors, namespace=namespace)
                success += len(vectors)
                by_namespace[namespace] = by_namespace.get(namespace, 0) + len(vectors)
            except Exception as exc:
                logger.exception("Upsert failed for namespace=%s batch_start=%s", namespace, start)
                for item in batch:
                    failed.append(
                        {
                            "id": item["id"],
                            "ticker": item["ticker"],
                            "namespace": namespace,
                            "error": str(exc),
                        }
                    )
        logger.info(
            "[%s] Successfully stored %s vectors this run.",
            namespace,
            by_namespace.get(namespace, 0),
        )

    return {
        "upserted_count": success,
        "failed_count": len(failed),
        "by_namespace": by_namespace,
        "failures": failed,
    }


def namespace_stats(index: Any) -> dict[str, int]:
    """Return vector counts per namespace from Pinecone index stats."""
    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    result: dict[str, int] = {}
    if not namespaces:
        return result
    for name, payload in namespaces.items():
        if hasattr(payload, "vector_count"):
            result[name] = int(payload.vector_count)
        elif isinstance(payload, dict):
            result[name] = int(payload.get("vector_count", 0))
    return result
