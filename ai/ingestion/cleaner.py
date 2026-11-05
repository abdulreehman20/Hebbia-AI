"""Light normalization and quality checks for loaded SEC Markdown documents."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document

from config import Settings, get_settings

logger = logging.getLogger(__name__)

REPLACEMENT_CHAR_RE = re.compile(r"\uFFFD+")
WEIRD_SEP_RE = re.compile(r"SeptemberA(?=\d)")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    """Normalize Unicode and common conversion artifacts without deleting numbers."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = REPLACEMENT_CHAR_RE.sub("", text)
    # Common mangled date artifact seen in Phase 1 HTML→MD conversion.
    text = WEIRD_SEP_RE.sub("September ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip() + "\n"


def document_quality_issues(document: Document, min_chars: int) -> list[str]:
    """Return quality issue codes for a document, if any."""
    issues: list[str] = []
    content = document.page_content or ""
    if len(content.strip()) < min_chars:
        issues.append("too_short")
    if not document.metadata.get("ticker"):
        issues.append("missing_ticker")
    if not document.metadata.get("accession_number"):
        issues.append("missing_accession")
    if not document.metadata.get("source_url"):
        issues.append("missing_source_url")
    return issues


def clean_documents(
    documents: Iterable[Document],
    settings: Settings | None = None,
) -> tuple[list[Document], list[dict[str, Any]], dict[str, Any]]:
    """Normalize documents and reject those that fail basic quality checks."""
    settings = settings or get_settings()
    cleaned: list[Document] = []
    rejected: list[dict[str, Any]] = []

    form_counts: dict[str, int] = {}
    sample_audit: list[dict[str, Any]] = []

    for document in documents:
        normalized = normalize_text(document.page_content)
        metadata = dict(document.metadata)
        metadata["cleaned_char_count"] = len(normalized)
        candidate = Document(page_content=normalized, metadata=metadata)

        issues = document_quality_issues(candidate, settings.min_document_chars)
        if issues:
            rejected.append(
                {
                    "ticker": metadata.get("ticker"),
                    "form": metadata.get("form"),
                    "source_document": metadata.get("source_document"),
                    "issues": issues,
                }
            )
            continue

        cleaned.append(candidate)
        form = str(metadata.get("form") or "unknown")
        form_counts[form] = form_counts.get(form, 0) + 1

    # Representative audit samples: prefer variety across companies/forms.
    seen_keys: set[str] = set()
    for document in cleaned:
        ticker = str(document.metadata.get("ticker") or "")
        form = str(document.metadata.get("form") or "")
        key = f"{ticker}:{form}"
        if key in seen_keys:
            continue
        if len(sample_audit) >= 16:
            break
        seen_keys.add(key)
        text = document.page_content
        sample_audit.append(
            {
                "ticker": ticker,
                "form": form,
                "source_document": document.metadata.get("source_document"),
                "char_count": len(text),
                "has_item_heading": bool(re.search(r"(?m)^##\s+Item\s+\d", text)),
                "has_financial_token": bool(
                    re.search(r"\$[\d,]+|\d+%|\bmillion\b|\bbillion\b", text, re.I)
                ),
                "has_table_row": "|" in text,
            }
        )

    audit = {
        "accepted_count": len(cleaned),
        "rejected_count": len(rejected),
        "form_counts": form_counts,
        "samples": sample_audit,
    }
    logger.info(
        "Document cleaning complete. accepted=%s rejected=%s",
        len(cleaned),
        len(rejected),
    )
    return cleaned, rejected, audit
