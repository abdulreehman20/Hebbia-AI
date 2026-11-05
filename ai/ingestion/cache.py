"""Simple fingerprint cache to skip re-embedding unchanged chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """JSONL cache keyed by chunk_id -> {fingerprint, values}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = row.get("chunk_id")
                if isinstance(chunk_id, str):
                    self._entries[chunk_id] = row

    def get(self, chunk_id: str, fingerprint: str) -> list[float] | None:
        row = self._entries.get(chunk_id)
        if not row:
            return None
        if row.get("fingerprint") != fingerprint:
            return None
        values = row.get("values")
        if not isinstance(values, list) or not values:
            return None
        return [float(v) for v in values]

    def put(self, chunk_id: str, fingerprint: str, values: list[float]) -> None:
        self._entries[chunk_id] = {
            "chunk_id": chunk_id,
            "fingerprint": fingerprint,
            "values": values,
            "dimension": len(values),
        }

    def flush(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in self._entries.values():
                handle.write(json.dumps(row) + "\n")
        temporary.replace(self.path)
        logger.info("Wrote embedding cache (%s entries) to %s", len(self._entries), self.path)
