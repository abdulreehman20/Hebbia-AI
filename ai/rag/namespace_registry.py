"""Company ticker to Pinecone namespace allowlist."""

from __future__ import annotations

from constants import COMPANY_NAMESPACES, SUPPORTED_TICKERS

# Re-export for call sites that import from the registry module.
__all__ = [
    "COMPANY_NAMESPACES",
    "SUPPORTED_TICKERS",
    "get_namespace",
    "validate_tickers",
    "list_supported_tickers",
]


def get_namespace(ticker: str) -> str:
    """Map an allowed ticker to its Pinecone namespace. Reject unknowns."""
    normalized = ticker.strip().upper()
    if normalized not in COMPANY_NAMESPACES:
        raise ValueError(
            f"Unsupported ticker '{ticker}'. Allowed: {sorted(SUPPORTED_TICKERS)}"
        )
    return COMPANY_NAMESPACES[normalized]


def validate_tickers(tickers: list[str]) -> list[str]:
    """Normalize and validate tickers against the eight-company allowlist."""
    if not tickers:
        raise ValueError("At least one company ticker is required")
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        value = ticker.strip().upper()
        if not value:
            continue
        if value not in SUPPORTED_TICKERS:
            raise ValueError(
                f"Unsupported ticker '{ticker}'. Allowed: {sorted(SUPPORTED_TICKERS)}"
            )
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one company ticker is required")
    return normalized


def list_supported_tickers() -> list[str]:
    return sorted(SUPPORTED_TICKERS)
