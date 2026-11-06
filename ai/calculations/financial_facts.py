"""Extract and validate financial facts from evidence text."""

from __future__ import annotations

import re
from decimal import Decimal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.calculations.normalization import parse_decimal
from ai.providers.openrouter_chat import get_chat_model
from ai.schemas.financial import FinancialFact, FinancialPeriod
from ai.schemas.retrieval import RetrievedChunk
from config import Settings, get_settings

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(\$?\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?)(?![A-Za-z0-9])"
)


class _ExtractedFact(BaseModel):
    metric_name: str
    value: str
    currency: str | None = "USD"
    unit: str | None = None
    scale: str | None = None
    fiscal_year: int | None = None
    period_label: str | None = None
    quotation: str


class _ExtractionPayload(BaseModel):
    facts: list[_ExtractedFact] = Field(default_factory=list)


class FinancialFactExtractor:
    """LLM-assisted extraction with quotation reconciliation against source text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def extract(
        self,
        chunks: list[RetrievedChunk],
        *,
        metrics: list[str] | None = None,
    ) -> list[FinancialFact]:
        facts: list[FinancialFact] = []
        for chunk in chunks:
            facts.extend(self._extract_chunk(chunk, metrics=metrics or []))
        return facts

    def _extract_chunk(
        self,
        chunk: RetrievedChunk,
        *,
        metrics: list[str],
    ) -> list[FinancialFact]:
        if not chunk.text.strip():
            return []
        try:
            model = get_chat_model(self.settings, temperature=0.0)
            structured = model.with_structured_output(_ExtractionPayload)
            system = SystemMessage(
                content=(
                    "Extract numerical financial facts from the SEC excerpt. "
                    "Only use numbers that appear verbatim in the text. "
                    "Include a short quotation containing each number. "
                    f"Preferred metrics: {metrics or ['any reported financial metrics']}."
                )
            )
            human = HumanMessage(
                content=f"Ticker: {chunk.ticker}\nChunk ID: {chunk.chunk_id}\n\n{chunk.text[:6000]}"
            )
            payload = structured.invoke([system, human])
            if isinstance(payload, dict):
                payload = _ExtractionPayload.model_validate(payload)
            assert isinstance(payload, _ExtractionPayload)
        except Exception:
            return self._regex_fallback(chunk, metrics=metrics)

        validated: list[FinancialFact] = []
        for item in payload.facts:
            if not self._quotation_supported(chunk.text, item.quotation, item.value):
                continue
            try:
                value = parse_decimal(item.value)
            except ValueError:
                continue
            validated.append(
                FinancialFact(
                    metric_name=item.metric_name,
                    value=value,
                    currency=item.currency,
                    unit=item.unit,
                    scale=item.scale,
                    period=FinancialPeriod(
                        fiscal_year=item.fiscal_year,
                        label=item.period_label,
                    ),
                    fiscal_year=item.fiscal_year,
                    ticker=chunk.ticker,
                    accession_number=chunk.accession_number,
                    chunk_id=chunk.chunk_id,
                    quotation=item.quotation,
                )
            )
        return validated

    def _regex_fallback(
        self,
        chunk: RetrievedChunk,
        *,
        metrics: list[str],
    ) -> list[FinancialFact]:
        """Conservative keyword+number extraction when LLM is unavailable."""
        text = chunk.text
        lower = text.lower()
        facts: list[FinancialFact] = []
        targets = metrics or ["revenue", "net income", "operating income"]
        for metric in targets:
            token = metric.replace("_", " ").lower()
            idx = lower.find(token)
            if idx < 0:
                continue
            window = text[idx : idx + 120]
            match = _NUMBER_RE.search(window)
            if not match:
                continue
            raw = match.group(1)
            try:
                value = parse_decimal(raw.replace("(", "-").replace(")", ""))
            except ValueError:
                continue
            facts.append(
                FinancialFact(
                    metric_name=metric.replace(" ", "_"),
                    value=value,
                    ticker=chunk.ticker,
                    accession_number=chunk.accession_number,
                    chunk_id=chunk.chunk_id,
                    quotation=window.strip()[:200],
                )
            )
        return facts

    @staticmethod
    def _quotation_supported(text: str, quotation: str, value: str) -> bool:
        if not quotation or quotation not in text:
            # Allow whitespace-normalized containment.
            compact_text = " ".join(text.split())
            compact_quote = " ".join(quotation.split())
            if compact_quote not in compact_text:
                return False
        digits = re.sub(r"[^\d.]", "", value)
        if not digits:
            return False
        return digits in re.sub(r"[^\d.]", " ", text)
