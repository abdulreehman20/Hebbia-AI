"""Extractive context compression."""

from __future__ import annotations

import re

from ai.schemas.retrieval import RetrievedChunk

_BOILERPLATE = re.compile(
    r"(table of contents|forward-looking statements|united states securities and exchange commission)",
    re.IGNORECASE,
)


def compress_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Dedupe by chunk_id and drop near-duplicate / boilerplate-only passages."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    out: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue
        normalized = " ".join(chunk.text.lower().split())
        if not normalized:
            continue
        if normalized in seen_text:
            continue
        if _BOILERPLATE.search(chunk.text) and len(normalized) < 400:
            # Keep short boilerplate only if it still has digits (possible tables).
            if not re.search(r"\d", chunk.text):
                continue
        seen_ids.add(chunk.chunk_id)
        seen_text.add(normalized)
        out.append(chunk)
    return out
