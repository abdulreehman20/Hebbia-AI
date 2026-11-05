"""Structure-aware chunking for SEC Markdown filings."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, get_settings
from constants import INGESTION_VERSION

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
ITEM_RE = re.compile(r"(?i)^Item\s+\d{1,2}[A-C]?(?:\.\d{2})?")


def approx_token_len(text: str) -> int:
    """Approximate token count (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


def _slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return (slug or "section")[:max_len]


def stable_chunk_id(
    ticker: str,
    accession_number: str,
    section: str,
    chunk_index: int,
    text: str,
) -> str:
    """Build a deterministic Pinecone-safe vector ID."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    accession = re.sub(r"[^0-9A-Za-z]+", "", accession_number or "unknown")
    return (
        f"{ticker.lower()}-{accession}-{_slug(section)}-{chunk_index}-{digest}"
    )


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_title, section_body) pairs."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [("Document", body)] if body else []

    sections: list[tuple[str, str]] = []
    # Preamble before the first heading.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body and not ITEM_RE.match(title):
            continue
        sections.append((title, body if body else title))
    return sections


def chunk_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    """Apply heading-aware then recursive splitting to cleaned documents."""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=approx_token_len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )

    chunks: list[Document] = []
    failed: list[dict[str, Any]] = []
    by_ticker_form: Counter[str] = Counter()
    created = 0

    logger.info("Starting document chunking...")
    for document in documents:
        ticker = str(document.metadata.get("ticker") or "UNK")
        form = str(document.metadata.get("form") or "UNK")
        accession = str(document.metadata.get("accession_number") or "")
        source_document = str(document.metadata.get("source_document") or "")

        try:
            sections = split_by_headings(document.page_content)
            doc_chunk_index = 0
            for section_title, section_body in sections:
                pieces = splitter.split_text(section_body)
                if not pieces and section_body.strip():
                    pieces = [section_body.strip()]
                for piece in pieces:
                    piece = piece.strip()
                    if not piece:
                        continue
                    chunk_id = stable_chunk_id(
                        ticker, accession, section_title, doc_chunk_index, piece
                    )
                    metadata = {
                        **document.metadata,
                        "section": section_title,
                        "chunk_index": doc_chunk_index,
                        "chunk_id": chunk_id,
                        "embedding_model": settings.openrouter_embedding_model,
                        "ingestion_version": INGESTION_VERSION,
                        "chunk_size_tokens": settings.chunk_size_tokens,
                        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
                        "source_document": source_document,
                    }
                    chunks.append(Document(page_content=piece, metadata=metadata))
                    doc_chunk_index += 1
                    created += 1
                    if created % 50 == 0:
                        logger.info("Created %s chunks...", created)

            by_ticker_form[f"{ticker}:{form}"] += doc_chunk_index
            logger.info(
                "[%s] Processed %s (%s chunks)",
                ticker,
                form,
                doc_chunk_index,
            )
        except Exception as exc:
            logger.exception("Failed chunking %s", source_document)
            failed.append(
                {
                    "ticker": ticker,
                    "source_document": source_document,
                    "error": str(exc),
                }
            )

    summary = {
        "documents_processed": len(documents) - len(failed),
        "documents_failed": len(failed),
        "total_chunks": len(chunks),
        "by_ticker_form": dict(sorted(by_ticker_form.items())),
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "failures": failed,
    }
    logger.info(
        "Chunking complete. documents=%s chunks=%s failed=%s",
        summary["documents_processed"],
        summary["total_chunks"],
        summary["documents_failed"],
    )
    return chunks, summary
