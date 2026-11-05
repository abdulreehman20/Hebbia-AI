"""Load Phase 1 Markdown filings into LangChain Documents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from langchain_core.documents import Document

from config import Settings, get_settings
from constants import SUPPORTED_TICKERS

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def resolve_markdown_path(relative_path: str, markdown_dir: Path) -> Path:
    """Resolve a manifest path while preventing escapes outside markdown/."""
    normalized = relative_path.replace("\\", "/")
    source_path = (markdown_dir / Path(*PurePosixPath(normalized).parts)).resolve()
    root = markdown_dir.resolve()
    if not source_path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes markdown directory: {relative_path}")
    return source_path


def parse_markdown_with_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML front matter from Markdown body."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    parsed = yaml.safe_load(raw_yaml) or {}
    if not isinstance(parsed, dict):
        raise ValueError("YAML front matter must be a mapping")
    body = text[match.end() :]
    return parsed, body


def _metadata_from_filing(filing: dict[str, Any], frontmatter: dict[str, Any]) -> dict[str, Any]:
    ticker = str(frontmatter.get("ticker") or filing.get("ticker") or "").upper()
    return {
        "ticker": ticker,
        "company_name": frontmatter.get("company_name") or filing.get("company_name"),
        "cik": str(frontmatter.get("cik") or filing.get("cik") or ""),
        "form": frontmatter.get("form") or filing.get("form"),
        "filing_date": frontmatter.get("filing_date") or filing.get("filing_date"),
        "report_date": frontmatter.get("report_date") or filing.get("report_date"),
        "accession_number": frontmatter.get("accession_number")
        or filing.get("accession_number"),
        "source_url": frontmatter.get("source_url") or filing.get("source_url"),
        "source_document": filing.get("markdown_local_path") or filing.get("local_path"),
        "source_local_path": filing.get("source_local_path"),
        "conversion_status": filing.get("conversion_status"),
    }


def load_documents(settings: Settings | None = None) -> tuple[list[Document], list[dict[str, Any]]]:
    """Load successfully converted Markdown filings from the Phase 1 manifest."""
    settings = settings or get_settings()
    manifest_path = settings.markdown_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Markdown manifest not found: {manifest_path}. Complete Phase 1 first."
        )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    filings = manifest.get("filings", [])
    if not isinstance(filings, list):
        raise ValueError("markdown/manifest.json must contain a 'filings' list")

    logger.info("Loading Markdown manifest...")
    logger.info("Found %s Markdown filing records.", len(filings))

    documents: list[Document] = []
    failures: list[dict[str, Any]] = []
    loaded_by_ticker: dict[str, int] = {}

    for filing in filings:
        if not isinstance(filing, dict):
            failures.append({"error": "non_object_filing", "filing": filing})
            continue

        status = filing.get("conversion_status")
        if status != "converted":
            failures.append(
                {
                    "ticker": filing.get("ticker"),
                    "source_document": filing.get("markdown_local_path"),
                    "error": f"skipped_status:{status}",
                }
            )
            continue

        relative = filing.get("markdown_local_path") or filing.get("local_path")
        ticker = str(filing.get("ticker") or "").upper()
        try:
            if not isinstance(relative, str) or not relative:
                raise ValueError("missing markdown_local_path")
            if ticker not in SUPPORTED_TICKERS:
                raise ValueError(f"unsupported ticker: {ticker}")

            path = resolve_markdown_path(relative, settings.markdown_dir)
            if not path.is_file():
                raise FileNotFoundError(f"Markdown file not found: {path}")

            raw = path.read_text(encoding="utf-8")
            frontmatter, body = parse_markdown_with_frontmatter(raw)
            metadata = _metadata_from_filing(filing, frontmatter)
            metadata["markdown_path"] = relative
            # Keep original raw text reference for provenance.
            metadata["raw_char_count"] = len(raw)
            documents.append(Document(page_content=body, metadata=metadata))
            loaded_by_ticker[ticker] = loaded_by_ticker.get(ticker, 0) + 1
        except Exception as exc:
            logger.exception("Failed loading %s", relative)
            failures.append(
                {
                    "ticker": ticker,
                    "source_document": relative,
                    "error": str(exc),
                }
            )

    for ticker in sorted(loaded_by_ticker):
        logger.info("[%s] Loaded %s documents.", ticker, loaded_by_ticker[ticker])

    logger.info("Total documents loaded: %s", len(documents))
    logger.info("Failed documents: %s", len(failures))
    return documents, failures
