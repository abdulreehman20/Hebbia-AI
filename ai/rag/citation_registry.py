"""Assign and validate stable citation identifiers for evidence chunks."""

from __future__ import annotations

from ai.schemas.citations import CitationValidationResult, SourceCitation
from ai.schemas.retrieval import RetrievedChunk


class CitationRegistry:
    """Track [S1], [S2], ... citations for retrieved chunks."""

    def __init__(self) -> None:
        self._citations: list[SourceCitation] = []
        self._by_chunk: dict[str, SourceCitation] = {}
        self._by_id: dict[str, SourceCitation] = {}

    def register(self, chunk: RetrievedChunk) -> SourceCitation:
        existing = self._by_chunk.get(chunk.chunk_id)
        if existing is not None:
            return existing
        index = len(self._citations) + 1
        citation_id = f"S{index}"
        label = self._label(citation_id, chunk)
        citation = SourceCitation(
            citation_id=citation_id,
            label=label,
            chunk_id=chunk.chunk_id,
            ticker=chunk.ticker,
            form=chunk.form,
            filing_date=chunk.filing_date,
            section=chunk.section,
            source_url=chunk.source_url,
        )
        self._citations.append(citation)
        self._by_chunk[chunk.chunk_id] = citation
        self._by_id[citation_id] = citation
        return citation

    def register_many(self, chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        return [self.register(chunk) for chunk in chunks]

    def get(self, citation_id: str) -> SourceCitation | None:
        return self._by_id.get(citation_id)

    def all(self) -> list[SourceCitation]:
        return list(self._citations)

    def validate_ids(self, citation_ids: list[str]) -> CitationValidationResult:
        invalid = [cid for cid in citation_ids if cid not in self._by_id]
        return CitationValidationResult(
            valid=not invalid,
            invalid_citation_ids=invalid,
            message=None if not invalid else f"Unknown citation ids: {invalid}",
        )

    @staticmethod
    def _label(citation_id: str, chunk: RetrievedChunk) -> str:
        company = chunk.company_name or chunk.ticker
        form = chunk.form or "filing"
        filed = f", filed {chunk.filing_date}" if chunk.filing_date else ""
        section = f", {chunk.section}" if chunk.section else ""
        return f"[{citation_id}] {company} {form}{filed}{section}"
