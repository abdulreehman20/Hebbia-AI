"""Shared constants for the financial research backend."""

from __future__ import annotations

COMPANY_NAMESPACES: dict[str, str] = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "GOOGL": "google",
    "AMZN": "amazon",
    "NVDA": "nvidia",
    "META": "meta",
    "WMT": "walmart",
    "ORCL": "oracle",
}

SUPPORTED_TICKERS: frozenset[str] = frozenset(COMPANY_NAMESPACES)

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "WMT": "Walmart Inc.",
    "ORCL": "Oracle Corporation",
}

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

INGESTION_VERSION = "phase2-v1"
DEFAULT_PINECONE_INDEX_NAME = "hebbia-ai"
