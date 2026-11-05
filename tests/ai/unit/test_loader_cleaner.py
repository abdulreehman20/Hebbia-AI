"""Unit tests for Markdown loading helpers and cleaning."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ai.ingestion.cleaner import normalize_text
from ai.ingestion.loader import parse_markdown_with_frontmatter, resolve_markdown_path


def test_parse_frontmatter() -> None:
    text = """---
ticker: NVDA
form: 10-K
source_url: "https://www.sec.gov/example"
---

## Item 1. Business

Hello.
"""
    meta, body = parse_markdown_with_frontmatter(text)
    assert meta["ticker"] == "NVDA"
    assert meta["form"] == "10-K"
    assert body.lstrip().startswith("## Item 1")


def test_resolve_markdown_path_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "markdown"
    root.mkdir()
    try:
        resolve_markdown_path("../secret.txt", root)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_normalize_text_fixes_september_artifact() -> None:
    cleaned = normalize_text("For the fiscal year ended SeptemberA25, 2021")
    assert "September 25, 2021" in cleaned
    assert "SeptemberA" not in cleaned


def test_normalize_preserves_financial_values() -> None:
    text = "Revenue was $123,456 million (42%)."
    assert "$123,456" in normalize_text(text)
    assert "42%" in normalize_text(text)
    assert isinstance(Document(page_content=text), Document)
