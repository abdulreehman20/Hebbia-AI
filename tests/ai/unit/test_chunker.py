"""Unit tests for structure-aware chunking helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from ai.ingestion.chunker import approx_token_len, chunk_documents, split_by_headings, stable_chunk_id


def test_split_by_headings_keeps_items() -> None:
    text = """Preamble text.

## Item 1. Business

Apple designs products.

## Item 1A. Risk Factors

Risks include competition.
"""
    sections = split_by_headings(text)
    titles = [title for title, _ in sections]
    assert "Preamble" in titles
    assert any(title.startswith("Item 1.") for title in titles)
    assert any("Risk Factors" in title for title in titles)


def test_stable_chunk_id_is_deterministic() -> None:
    first = stable_chunk_id("NVDA", "0001045810-25-000023", "Item 7", 0, "revenue grew")
    second = stable_chunk_id("NVDA", "0001045810-25-000023", "Item 7", 0, "revenue grew")
    assert first == second
    assert first.startswith("nvda-")


def test_chunk_documents_assigns_metadata() -> None:
    document = Document(
        page_content=(
            "## Item 7. MD&A\n\n"
            + ("Revenue increased substantially. " * 80)
            + "\n\n## Item 8. Financial Statements\n\n"
            + ("Net income was higher year over year. " * 80)
        ),
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "cik": "0000320193",
            "form": "10-K",
            "filing_date": "2025-10-31",
            "report_date": "2025-09-27",
            "accession_number": "0000320193-25-000001",
            "source_url": "https://www.sec.gov/example",
            "source_document": "AAPL/10-K/2025/example.md",
        },
    )
    chunks, summary = chunk_documents([document])
    assert summary["total_chunks"] >= 2
    assert all(chunk.metadata.get("chunk_id") for chunk in chunks)
    assert all(chunk.metadata.get("section") for chunk in chunks)
    assert all(chunk.metadata.get("ticker") == "AAPL" for chunk in chunks)
    assert approx_token_len(chunks[0].page_content) > 0
