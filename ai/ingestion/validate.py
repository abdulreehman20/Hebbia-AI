"""Post-ingestion Pinecone validation and retrieval smoke tests."""

from __future__ import annotations

import json
import logging
from typing import Any

from ai.ingestion.pinecone_store import _client, get_namespace, namespace_stats
from ai.providers.openrouter_embeddings import OpenRouterEmbeddings
from config import Settings, get_settings
from constants import COMPANY_NAMESPACES

logger = logging.getLogger(__name__)

RETRIEVAL_QUERIES: dict[str, str] = {
    "NVDA": "What were NVIDIA's main sources of revenue in fiscal 2025?",
    "MSFT": "What are Microsoft's major business segments and how do they contribute to revenue?",
    "AAPL": "What were Apple's primary revenue drivers and product categories?",
    "AMZN": "How does Amazon describe its revenue segments and operating income?",
    "GOOGL": "What are Alphabet's major business segments and advertising revenue?",
    "META": "What are Meta's main revenue sources and family of apps?",
    "WMT": "How does Walmart describe its business segments and net sales?",
    "ORCL": "What are Oracle's primary cloud and software revenue drivers?",
}


def validate_ingestion(
    report: dict[str, Any] | None = None,
    settings: Settings | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Verify namespaces, counts, and company-scoped dense retrieval."""
    settings = settings or get_settings()
    settings.require_openrouter()
    settings.require_pinecone()

    if report is None:
        if not settings.ingestion_report_path.is_file():
            raise FileNotFoundError(
                f"Ingestion report not found: {settings.ingestion_report_path}"
            )
        report = json.loads(settings.ingestion_report_path.read_text(encoding="utf-8"))

    pc = _client(settings)
    index = pc.Index(settings.pinecone_index_name)
    stats = namespace_stats(index)
    embeddings = OpenRouterEmbeddings(settings)

    expected = sorted(COMPANY_NAMESPACES.values())
    missing_namespaces = [name for name in expected if stats.get(name, 0) <= 0]

    retrieval_results: dict[str, Any] = {}
    for ticker, question in RETRIEVAL_QUERIES.items():
        namespace = get_namespace(ticker)
        vector = embeddings.embed_query(question)
        response = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        simplified = []
        cross_company = False
        for match in matches or []:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata", {})
            metadata = metadata or {}
            match_ticker = str(metadata.get("ticker") or "").upper()
            if match_ticker and match_ticker != ticker:
                cross_company = True
            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score")
            match_id = getattr(match, "id", None)
            if match_id is None and isinstance(match, dict):
                match_id = match.get("id")
            simplified.append(
                {
                    "id": match_id,
                    "score": score,
                    "ticker": match_ticker,
                    "form": metadata.get("form"),
                    "filing_date": metadata.get("filing_date"),
                    "section": metadata.get("section"),
                    "source_document": metadata.get("source_document"),
                    "source_url": metadata.get("source_url"),
                    "text_preview": str(metadata.get("text") or "")[:240],
                }
            )

        retrieval_results[ticker] = {
            "namespace": namespace,
            "question": question,
            "match_count": len(simplified),
            "cross_company_contamination": cross_company,
            "matches": simplified,
        }
        logger.info(
            "[%s] retrieval matches=%s cross_company=%s",
            ticker,
            len(simplified),
            cross_company,
        )

    validation = {
        "pinecone_index": settings.pinecone_index_name,
        "vector_dimensions": report.get("vector_dimensions"),
        "namespace_counts": stats,
        "missing_namespaces": missing_namespaces,
        "upserted_count": report.get("upserts", {}).get("upserted_count"),
        "embedding_success_count": report.get("embeddings", {}).get("successful_embeddings"),
        "retrieval": retrieval_results,
        "all_namespaces_present": not missing_namespaces,
        "retrieval_ok": all(
            item["match_count"] > 0 and not item["cross_company_contamination"]
            for item in retrieval_results.values()
        ),
    }

    report["validation"] = validation
    settings.ingestion_report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Validation complete. namespaces_ok=%s retrieval_ok=%s",
                validation["all_namespaces_present"], validation["retrieval_ok"])
    return validation
