"""End-to-end Phase 2 ingestion orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from ai.ingestion.cache import EmbeddingCache
from ai.ingestion.chunker import chunk_documents
from ai.ingestion.cleaner import clean_documents
from ai.ingestion.loader import load_documents
from ai.ingestion.pinecone_store import ensure_index, namespace_stats, upsert_chunks
from ai.providers.openrouter_embeddings import OpenRouterEmbeddings, RateLimitError
from config import Settings, get_settings
from constants import COMPANY_NAMESPACES, INGESTION_VERSION

logger = logging.getLogger(__name__)


def _fingerprint(text: str, model: str) -> str:
    digest = hashlib.sha1(f"{model}\n{text}".encode("utf-8")).hexdigest()
    return digest


def embed_chunks(
    chunks: list[Document],
    embeddings: OpenRouterEmbeddings,
    cache: EmbeddingCache,
    settings: Settings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed chunks with fingerprint cache and free-tier rate-limit pacing."""
    logger.info("Starting embedding generation...")
    logger.info("Model: %s", settings.openrouter_embedding_model)
    logger.info(
        "Total chunks: %s (batch_size=%s pause=%.1fs)",
        len(chunks),
        settings.embedding_batch_size,
        settings.embedding_batch_pause_seconds,
    )

    failed: list[dict[str, Any]] = []
    cache_hits = 0
    embedded_now = 0
    consecutive_rate_limits = 0
    stopped_early = False

    pending_indices: list[int] = []
    pending_texts: list[str] = []
    pending_meta: list[tuple[str, str, Document]] = []

    prepared: list[dict[str, Any] | None] = [None] * len(chunks)

    for index, chunk in enumerate(chunks):
        chunk_id = str(chunk.metadata.get("chunk_id"))
        fp = _fingerprint(chunk.page_content, settings.openrouter_embedding_model)
        cached = cache.get(chunk_id, fp)
        if cached is not None:
            prepared[index] = {
                "id": chunk_id,
                "values": cached,
                "metadata": chunk.metadata,
                "text": chunk.page_content,
                "ticker": chunk.metadata.get("ticker"),
            }
            cache_hits += 1
            continue
        pending_indices.append(index)
        pending_texts.append(chunk.page_content)
        pending_meta.append((chunk_id, fp, chunk))

    logger.info(
        "Cache hits: %s. Remaining to embed this run: %s",
        cache_hits,
        len(pending_texts),
    )

    batch_size = max(1, settings.embedding_batch_size)
    try:
        for start in range(0, len(pending_texts), batch_size):
            batch_texts = pending_texts[start : start + batch_size]
            batch_meta = pending_meta[start : start + batch_size]
            batch_indices = pending_indices[start : start + batch_size]
            try:
                # One OpenRouter request per pipeline batch (already sized).
                vectors = embeddings.embed_batch(batch_texts)
                for offset, vector in enumerate(vectors):
                    chunk_id, fp, chunk = batch_meta[offset]
                    cache.put(chunk_id, fp, vector)
                    prepared[batch_indices[offset]] = {
                        "id": chunk_id,
                        "values": vector,
                        "metadata": chunk.metadata,
                        "text": chunk.page_content,
                        "ticker": chunk.metadata.get("ticker"),
                    }
                    embedded_now += 1
                # Persist after every success so Ctrl+C / crashes keep progress.
                cache.flush()
                consecutive_rate_limits = 0
                done = cache_hits + embedded_now
                logger.info("Generated embeddings: %s / %s", done, len(chunks))
                if (
                    start + batch_size < len(pending_texts)
                    and settings.embedding_batch_pause_seconds > 0
                ):
                    time.sleep(settings.embedding_batch_pause_seconds)
            except RateLimitError as exc:
                consecutive_rate_limits += 1
                logger.exception("Rate-limited embedding batch at offset %s", start)
                cache.flush()
                for chunk_id, _fp, chunk in batch_meta:
                    failed.append(
                        {
                            "chunk_id": chunk_id,
                            "ticker": chunk.metadata.get("ticker"),
                            "source_document": chunk.metadata.get("source_document"),
                            "error": str(exc),
                        }
                    )
                if consecutive_rate_limits >= settings.max_consecutive_rate_limits:
                    remaining = len(pending_texts) - start - len(batch_texts)
                    stopped_early = True
                    logger.warning(
                        "Stopping embedding early after %s consecutive rate limits. "
                        "Cache flushed. Re-run later to resume (%s chunks still pending).",
                        consecutive_rate_limits,
                        remaining,
                    )
                    break
                # Cool down before trying the next batch on free tier.
                cool_down = min(
                    30.0 * consecutive_rate_limits,
                    settings.rate_limit_max_backoff_seconds,
                )
                logger.info("Cooling down %.1fs before next batch...", cool_down)
                time.sleep(cool_down)
            except Exception as exc:
                consecutive_rate_limits = 0
                logger.exception("Embedding batch failed at offset %s", start)
                cache.flush()
                for chunk_id, _fp, chunk in batch_meta:
                    failed.append(
                        {
                            "chunk_id": chunk_id,
                            "ticker": chunk.metadata.get("ticker"),
                            "source_document": chunk.metadata.get("source_document"),
                            "error": str(exc),
                        }
                    )
    except KeyboardInterrupt:
        logger.warning("Interrupted — flushing embedding cache before exit.")
        cache.flush()
        raise
    finally:
        cache.flush()

    records = [row for row in prepared if row is not None]
    summary = {
        "total_chunks": len(chunks),
        "successful_embeddings": len(records),
        "failed_embeddings": len(failed),
        "cache_hits": cache_hits,
        "embedded_this_run": embedded_now,
        "pending_remaining": max(0, len(chunks) - len(records)),
        "stopped_early_rate_limit": stopped_early,
        "vector_dimensions": embeddings.dimension,
        "failures": failed,
    }
    logger.info(
        "Embedding generation completed. success=%s failed=%s dim=%s stopped_early=%s",
        summary["successful_embeddings"],
        summary["failed_embeddings"],
        summary["vector_dimensions"],
        stopped_early,
    )
    return records, summary


def run_ingestion(settings: Settings | None = None) -> dict[str, Any]:
    """Execute load → clean → chunk → embed → upsert and return a report."""
    settings = settings or get_settings()
    settings.require_openrouter()
    settings.require_pinecone()
    settings.ingestion_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting SEC document ingestion...")
    documents, load_failures = load_documents(settings)
    cleaned, rejected, audit = clean_documents(documents, settings)
    chunks, chunk_summary = chunk_documents(cleaned, settings)

    embeddings = OpenRouterEmbeddings(settings)
    dimension = embeddings.probe_dimension()
    logger.info("Measured embedding dimension: %s", dimension)

    cache = EmbeddingCache(settings.ingestion_cache_dir / "embeddings.jsonl")
    records, embed_summary = embed_chunks(chunks, embeddings, cache, settings)

    if embed_summary.get("stopped_early_rate_limit"):
        logger.warning(
            "Partial embedding run due to free-tier rate limits. "
            "Upserting %s available vectors; re-run ingestion to resume remaining chunks.",
            len(records),
        )

    index = ensure_index(dimension, settings)
    upsert_summary = upsert_chunks(index, records, settings)
    stats = namespace_stats(index)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "ingestion_version": INGESTION_VERSION,
        "embedding_model": settings.openrouter_embedding_model,
        "vector_dimensions": dimension,
        "pinecone_index": settings.pinecone_index_name,
        "company_namespaces": COMPANY_NAMESPACES,
        "source_markdown_documents": len(documents) + len(load_failures),
        "loaded_documents": len(documents),
        "load_failures": load_failures,
        "rejected_during_validation": rejected,
        "quality_audit": audit,
        "chunking": chunk_summary,
        "embeddings": embed_summary,
        "upserts": upsert_summary,
        "pinecone_namespace_counts": stats,
        "expected_namespaces": sorted(COMPANY_NAMESPACES.values()),
    }

    settings.ingestion_report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.ingestion_report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(settings.ingestion_report_path)
    logger.info("Wrote ingestion report to %s", settings.ingestion_report_path)
    return report
