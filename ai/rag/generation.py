"""Grounded answer generation from an EvidencePackage."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ai.context.builder import ContextBuilder
from ai.providers.openrouter_chat import get_chat_model
from ai.rag.prompts import ANSWER_INSTRUCTION, SYSTEM_PROMPT
from ai.schemas.answers import ResearchAnswer
from ai.schemas.evidence import EvidencePackage
from ai.schemas.plan import ResearchPlan
from ai.schemas.research import ResearchRequest
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.context_builder = ContextBuilder(self.settings)

    def generate(
        self,
        request: ResearchRequest,
        plan: ResearchPlan | None,
        package: EvidencePackage,
    ) -> ResearchAnswer:
        if not package.evidence_items and not package.calculations:
            return ResearchAnswer(
                executive_summary=(
                    "I could not find sufficient evidence in the available SEC filings "
                    "to answer this question."
                ),
                key_findings=[],
                financial_analysis=None,
                calculations=[],
                company_comparisons=None,
                risks_and_limitations=[
                    "Retrieval returned no usable evidence chunks.",
                    "Pinecone ingest may still be incomplete.",
                ],
                missing_information=[
                    "Requested financial evidence was not found in the research database."
                ],
                citations=[],
                insufficient_evidence=True,
            )

        context_text = self.context_builder.render_for_prompt(package)
        try:
            model = get_chat_model(self.settings, temperature=0.1)
            structured = model.with_structured_output(ResearchAnswer)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"{ANSWER_INSTRUCTION}\n\n"
                        f"User question: {request.question}\n"
                        f"Plan: {plan.model_dump_json() if plan else '{}'}\n\n"
                        f"{context_text}\n\n"
                        f"Allowed citations: "
                        f"{[c.model_dump() for c in package.citations]}"
                    )
                ),
            ]
            result = structured.invoke(messages)
            if isinstance(result, dict):
                answer = ResearchAnswer.model_validate(result)
            else:
                answer = result
            return self._sanitize(answer, package)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structured answer generation failed: %s", exc)
            return self._fallback_answer(package)

    def _sanitize(self, answer: ResearchAnswer, package: EvidencePackage) -> ResearchAnswer:
        allowed = {c.citation_id for c in package.citations}
        citations = [c for c in answer.citations if c.citation_id in allowed]
        if not citations:
            citations = list(package.citations)
        # Never invent calculations — prefer engine results.
        calculations = package.calculations or answer.calculations
        insufficient = answer.insufficient_evidence or not package.evidence_items
        return answer.model_copy(
            update={
                "citations": citations,
                "calculations": calculations,
                "insufficient_evidence": insufficient,
            }
        )

    @staticmethod
    def _fallback_answer(package: EvidencePackage) -> ResearchAnswer:
        findings = []
        for item in package.evidence_items[:5]:
            snippet = item.chunk.text.strip().replace("\n", " ")
            findings.append(f"[{item.citation_id}] {snippet[:240]}")
        return ResearchAnswer(
            executive_summary=(
                "Retrieved evidence is summarized below. Structured generation was unavailable."
            ),
            key_findings=findings,
            financial_analysis=None,
            calculations=package.calculations,
            company_comparisons=None,
            risks_and_limitations=["Answer generated without LLM structured output."],
            missing_information=[m.description for m in package.missing],
            citations=package.citations,
            insufficient_evidence=not package.evidence_items,
            raw_markdown=None,
        )
