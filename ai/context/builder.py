"""Build EvidencePackage for answer generation."""

from __future__ import annotations

from ai.context.compressor import compress_chunks
from ai.context.prioritizer import prioritize_chunks
from ai.context.token_budget import TokenBudget, estimate_tokens
from ai.rag.citation_registry import CitationRegistry
from ai.schemas.evidence import EvidenceItem, EvidencePackage, EvidenceSource, MissingEvidence
from ai.schemas.financial import CalculationResult, FinancialFact
from ai.schemas.plan import ResearchPlan
from ai.schemas.research import ResearchIntent
from ai.schemas.retrieval import RetrievedChunk
from config import Settings, get_settings


class ContextBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build(
        self,
        *,
        question: str,
        intent: ResearchIntent | None = None,
        plan: ResearchPlan | None = None,
        chunks: list[RetrievedChunk],
        facts: list[FinancialFact] | None = None,
        calculations: list[CalculationResult] | None = None,
        missing: list[MissingEvidence] | None = None,
        conflicts: list[str] | None = None,
        registry: CitationRegistry | None = None,
    ) -> EvidencePackage:
        registry = registry or CitationRegistry()
        ordered = prioritize_chunks(chunks, intent)
        ordered = compress_chunks(ordered)

        budget = TokenBudget(self.settings.context_token_budget)
        # Reserve headroom for system + answer instructions.
        budget.consume("x" * 800)
        selected: list[RetrievedChunk] = []
        for chunk in ordered:
            block = f"[{chunk.ticker}] {chunk.form} {chunk.section}\n{chunk.text}"
            if not budget.fits(block):
                continue
            budget.consume(block)
            selected.append(chunk)

        citations = registry.register_many(selected)
        evidence_items: list[EvidenceItem] = []
        for chunk, citation in zip(selected, citations, strict=True):
            evidence_items.append(
                EvidenceItem(
                    citation_id=citation.citation_id,
                    chunk=chunk,
                    source=EvidenceSource(
                        chunk_id=chunk.chunk_id,
                        ticker=chunk.ticker,
                        form=chunk.form,
                        filing_date=chunk.filing_date,
                        accession_number=chunk.accession_number,
                        source_url=chunk.source_url,
                        section=chunk.section,
                    ),
                )
            )

        periods: list[str] = []
        if intent and intent.start_year and intent.end_year:
            periods.append(f"{intent.start_year}-{intent.end_year}")
        elif intent and intent.start_year:
            periods.append(str(intent.start_year))

        return EvidencePackage(
            question=question,
            objective=plan.objective if plan else None,
            companies=(plan.companies if plan else (intent.tickers if intent else [])),
            reporting_periods=periods,
            plan=plan,
            evidence_items=evidence_items,
            financial_facts=facts or [],
            calculations=calculations or [],
            citations=registry.all(),
            missing=missing or [],
            conflicts=conflicts or [],
        )

    def render_for_prompt(self, package: EvidencePackage) -> str:
        parts: list[str] = [
            f"Question: {package.question}",
            f"Objective: {package.objective or 'n/a'}",
            f"Companies: {', '.join(package.companies) or 'n/a'}",
            f"Periods: {', '.join(package.reporting_periods) or 'n/a'}",
            "",
            "Evidence:",
        ]
        for item in package.evidence_items:
            citation = next(
                (c for c in package.citations if c.citation_id == item.citation_id),
                None,
            )
            label = citation.label if citation else f"[{item.citation_id}]"
            parts.append(f"{label}\n{item.chunk.text}\n")
        if package.financial_facts:
            parts.append("Extracted facts:")
            for fact in package.financial_facts:
                parts.append(
                    f"- {fact.ticker} {fact.metric_name}={fact.value} "
                    f"(chunk {fact.chunk_id})"
                )
        if package.calculations:
            parts.append("Calculations:")
            for calc in package.calculations:
                parts.append(
                    f"- {calc.metric_name}: {calc.result} ({calc.status}) "
                    f"formula={calc.formula}"
                )
        if package.missing:
            parts.append("Missing information:")
            for miss in package.missing:
                parts.append(f"- {miss.description}")
        text = "\n".join(parts)
        # Soft truncate if somehow over budget.
        if estimate_tokens(text) > self.settings.context_token_budget:
            return text[: self.settings.context_token_budget * 4]
        return text
