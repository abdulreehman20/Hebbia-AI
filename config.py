"""Centralized application settings for AI and ingestion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import (
    CHAT_MODEL,
    DEFAULT_PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    OPENROUTER_EMBEDDINGS_URL,
    RERANKER_MODEL,
)

BACKEND_ROOT = Path(__file__).resolve().parent
DATA_DIR = BACKEND_ROOT / "data"
MARKDOWN_DIR = DATA_DIR / "markdown"
MARKDOWN_MANIFEST = MARKDOWN_DIR / "manifest.json"
INGESTION_CACHE_DIR = DATA_DIR / "ingestion_cache"
INGESTION_REPORT_PATH = DATA_DIR / "ingestion_report.json"
BM25_CACHE_PATH = DATA_DIR / "ingestion_cache" / "bm25_encoder.json"


class Settings(BaseSettings):
    """Typed settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_chat_model: str = Field(default=CHAT_MODEL, alias="OPENROUTER_CHAT_MODEL")
    openrouter_embedding_model: str = Field(
        default=EMBEDDING_MODEL,
        alias="OPENROUTER_EMBEDDING_MODEL",
    )
    openrouter_embeddings_url: str = Field(
        default=OPENROUTER_EMBEDDINGS_URL,
        alias="OPENROUTER_EMBEDDINGS_URL",
    )
    chat_temperature: float = Field(default=0.1, alias="CHAT_TEMPERATURE")
    chat_max_tokens: int = Field(default=4096, alias="CHAT_MAX_TOKENS")

    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default=DEFAULT_PINECONE_INDEX_NAME,
        alias="PINECONE_INDEX_NAME",
    )
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    chunk_size_tokens: int = Field(default=1000, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=150, alias="CHUNK_OVERLAP_TOKENS")
    embedding_batch_size: int = Field(default=8, alias="EMBEDDING_BATCH_SIZE")
    embedding_batch_pause_seconds: float = Field(
        default=1.5,
        alias="EMBEDDING_BATCH_PAUSE_SECONDS",
    )
    max_consecutive_rate_limits: int = Field(
        default=3,
        alias="MAX_CONSECUTIVE_RATE_LIMITS",
    )
    rate_limit_max_backoff_seconds: float = Field(
        default=120.0,
        alias="RATE_LIMIT_MAX_BACKOFF_SECONDS",
    )
    pinecone_upsert_batch_size: int = Field(default=100, alias="PINECONE_UPSERT_BATCH_SIZE")

    agent_max_iterations: int = Field(default=6, alias="AGENT_MAX_ITERATIONS")
    retrieval_top_k: int = Field(default=30, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=7, alias="RERANK_TOP_N")
    context_token_budget: int = Field(default=20000, alias="CONTEXT_TOKEN_BUDGET")
    max_query_rewrites: int = Field(default=4, alias="MAX_QUERY_REWRITES")
    max_retrieval_calls: int = Field(default=8, alias="MAX_RETRIEVAL_CALLS")
    reranker_model: str = Field(default=RERANKER_MODEL, alias="RERANKER_MODEL")

    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(default=8, alias="MAX_RETRIES")
    min_document_chars: int = Field(default=200, alias="MIN_DOCUMENT_CHARS")

    markdown_dir: Path = Field(default=MARKDOWN_DIR)
    markdown_manifest: Path = Field(default=MARKDOWN_MANIFEST)
    ingestion_cache_dir: Path = Field(default=INGESTION_CACHE_DIR)
    ingestion_report_path: Path = Field(default=INGESTION_REPORT_PATH)
    bm25_cache_path: Path = Field(default=BM25_CACHE_PATH)

    def require_openrouter(self) -> None:
        if not self.openrouter_api_key.strip():
            raise ValueError(
                "OPENROUTER_API_KEY is required. Set it in backend/.env before running AI features."
            )

    def require_pinecone(self) -> None:
        if not self.pinecone_api_key.strip():
            raise ValueError(
                "PINECONE_API_KEY is required. Set it in backend/.env before running AI features."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
