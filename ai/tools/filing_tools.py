"""Filing context expansion and amendment helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai.providers.openrouter_embeddings import OpenRouterEmbeddings
from ai.rag.context_expansion import ContextExpansionService
from ai.rag.hybrid_retriever import HybridRetriever
from ai.schemas.retrieval import RetrievedChunk, RetrievalFilters


class _ExpandArgs(BaseModel):
    ticker: str
    chunk_id: str
    text: str
    accession_number: str | None = None
    chunk_index: int | None = None
    namespace: str | None = None
    window: int = 1


@tool("expand_document_context", args_schema=_ExpandArgs)
def expand_document_context(
    ticker: str,
    chunk_id: str,
    text: str,
    accession_number: str | None = None,
    chunk_index: int | None = None,
    namespace: str | None = None,
    window: int = 1,
) -> dict[str, Any]:
    """Retrieve neighboring chunks around a selected evidence chunk."""
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        ticker=ticker,
        accession_number=accession_number,
        chunk_index=chunk_index,
        namespace=namespace,
    )
    embeddings = OpenRouterEmbeddings()
    vector = embeddings.embed_query(text[:1000] or ticker)
    neighbors = ContextExpansionService().expand(chunk, window=window, query_vector=vector)
    return {"chunks": [item.model_dump() for item in neighbors]}


class _AmendmentArgs(BaseModel):
    ticker: str
    query: str = Field(default="amendment restatement")
    filing_types: list[str] = Field(default_factory=lambda: ["10-K/A", "10-Q/A", "8-K/A"])


@tool("find_related_amendments", args_schema=_AmendmentArgs)
def find_related_amendments(
    ticker: str,
    query: str = "amendment restatement",
    filing_types: list[str] | None = None,
) -> dict[str, Any]:
    """Identify amended filing evidence related to the company."""
    filters = RetrievalFilters(forms=filing_types or ["10-K/A", "10-Q/A", "8-K/A"])
    result = HybridRetriever().retrieve(ticker=ticker, query=query, filters=filters)
    return {"ticker": ticker, "chunks": [c.model_dump() for c in result.chunks[:10]]}
