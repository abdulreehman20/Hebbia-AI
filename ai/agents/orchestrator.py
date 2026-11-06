"""Bounded single-agent Financial Research orchestration (no LangGraph)."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai.agents.planner import ResearchPlanner
from ai.agents.policies import AgentPolicies
from ai.agents.state import ResearchState
from ai.agents.tool_executor import ToolExecutor
from ai.calculations.engine import CalculationEngine
from ai.calculations.financial_facts import FinancialFactExtractor
from ai.context.builder import ContextBuilder
from ai.providers.openrouter_chat import get_chat_model
from ai.rag.citation_registry import CitationRegistry
from ai.rag.generation import AnswerGenerator
from ai.rag.prompts import SYSTEM_PROMPT
from ai.rag.query_understanding import QueryUnderstandingService
from ai.rag.retrieval_service import RetrievalService
from ai.schemas.evidence import MissingEvidence
from ai.schemas.research import ResearchRequest
from ai.schemas.retrieval import RetrievedChunk
from ai.tools.registry import get_research_tools
from config import Settings, get_settings

logger = logging.getLogger(__name__)


class FinancialResearchAgent:
    """Single Financial Research Agent with a bounded tool-calling loop."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.understanding = QueryUnderstandingService(self.settings)
        self.planner = ResearchPlanner()
        self.retrieval = RetrievalService(self.settings)
        self.policies = AgentPolicies(self.settings)
        self.executor = ToolExecutor(self.policies)
        self.context_builder = ContextBuilder(self.settings)
        self.generator = AnswerGenerator(self.settings)
        self.fact_extractor = FinancialFactExtractor(self.settings)
        self.calc_engine = CalculationEngine()

    def run(self, request: ResearchRequest) -> ResearchState:
        state = ResearchState(request_id=str(uuid.uuid4()), request=request, status="started")
        state.emit("research_started", request_id=state.request_id)
        try:
            self._execute(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Research agent failed")
            state.status = "failed"
            state.error = str(exc)
            state.emit("research_failed", error=str(exc))
        return state

    def stream_events(self, request: ResearchRequest) -> Iterator[dict[str, Any]]:
        state = self.run(request)
        for event in state.events:
            yield event
        yield {
            "event": "answer_completed",
            "request_id": state.request_id,
            "status": state.status,
            "answer": state.answer.model_dump(mode="json") if state.answer else None,
            "error": state.error,
        }

    def _execute(self, state: ResearchState) -> None:
        state.status = "planning"
        intent = self.understanding.understand(state.request)
        state.intent = intent
        if intent.task_type == "unsupported" or not intent.tickers:
            state.status = "unsupported"
            reason = intent.unsupported_reason or (
                "Supported companies only: AAPL, MSFT, GOOGL, AMZN, NVDA, META, WMT, ORCL."
            )
            state.missing.append(MissingEvidence(description=reason, tickers=intent.tickers))
            state.answer = self.generator.generate(
                state.request,
                None,
                self.context_builder.build(
                    question=state.request.question,
                    intent=intent,
                    chunks=[],
                    missing=state.missing,
                ),
            )
            state.emit("planning_completed", unsupported=True, reason=reason)
            state.emit("validation_completed", insufficient_evidence=True)
            state.status = "completed"
            return

        plan = self.planner.plan(state.request, intent)
        state.plan = plan
        state.emit(
            "planning_completed",
            task_type=plan.task_type,
            companies=plan.companies,
            operations=plan.operations,
        )

        state.status = "retrieving"
        # Deterministic retrieval path first (reliable without depending on free-tier tools).
        reranked = self.retrieval.retrieve_for_intent(state.request.question, intent)
        state.chunks = reranked.chunks
        state.emit(
            "retrieval_completed",
            chunk_count=len(state.chunks),
            hybrid_enabled=self.retrieval.retriever.hybrid_enabled,
            index_metric=self.retrieval.retriever.index_metric,
        )
        state.emit("reranking_completed", chunk_count=len(state.chunks))

        # Optional tool-calling refinement within iteration budget.
        self._tool_loop(state)

        state.status = "analyzing"
        if state.chunks and plan.requires_calculation:
            facts = self.fact_extractor.extract(state.chunks, metrics=plan.metrics)
            state.financial_facts = facts
            for metric in plan.metrics:
                if metric in self.calc_engine.supported_metrics:
                    # Only auto-calc when the caller supplied enough named facts via tools;
                    # otherwise leave calculations to explicit tool results already on state.
                    pass
        state.emit(
            "calculations_completed",
            fact_count=len(state.financial_facts),
            calculation_count=len(state.calculations),
        )

        registry = CitationRegistry()
        package = self.context_builder.build(
            question=state.request.question,
            intent=intent,
            plan=plan,
            chunks=state.chunks,
            facts=state.financial_facts,
            calculations=state.calculations,
            missing=state.missing,
            registry=registry,
        )
        state.citations = package.citations

        state.status = "generating"
        answer = self.generator.generate(state.request, plan, package)
        state.answer = answer
        state.emit(
            "validation_completed",
            insufficient_evidence=answer.insufficient_evidence,
            citation_count=len(answer.citations),
        )
        state.status = "completed"

    def _tool_loop(self, state: ResearchState) -> None:
        tools = get_research_tools()
        try:
            model = get_chat_model(self.settings, temperature=0.1)
            model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool binding unavailable; skipping tool loop: %s", exc)
            return

        messages: list[Any] = [
            SystemMessage(
                content=(
                    SYSTEM_PROMPT
                    + "\nUse only registered tools. Prefer retrieval tools for missing evidence. "
                    "Stop when you have enough evidence or hit limits."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "question": state.request.question,
                        "plan": state.plan.model_dump() if state.plan else {},
                        "retrieved_chunk_ids": [c.chunk_id for c in state.chunks[:10]],
                    }
                )
            ),
        ]

        for iteration in range(self.settings.agent_max_iterations):
            if not self.policies.allow_iteration(iteration):
                break
            try:
                ai_message: AIMessage = model_with_tools.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool loop model call failed: %s", exc)
                break
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                break
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
                record = self.executor.execute(name, args or {})
                state.tool_history.append(record)
                self._ingest_tool_result(state, name, record.result_preview)
                messages.append(
                    ToolMessage(
                        content=record.result_preview or "",
                        tool_call_id=call_id or name,
                    )
                )

    def _ingest_tool_result(self, state: ResearchState, name: str, preview: str | None) -> None:
        if not preview:
            return
        try:
            payload = json.loads(preview)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if "chunks" in payload and isinstance(payload["chunks"], list):
            for raw in payload["chunks"]:
                try:
                    chunk = RetrievedChunk.model_validate(raw)
                except Exception:  # noqa: BLE001
                    continue
                if all(existing.chunk_id != chunk.chunk_id for existing in state.chunks):
                    state.chunks.append(chunk)
        if name == "calculate_financial_metrics" and payload.get("status"):
            try:
                from ai.schemas.financial import CalculationResult

                state.calculations.append(CalculationResult.model_validate(payload))
            except Exception:  # noqa: BLE001
                return
        if name == "extract_financial_facts" and "facts" in payload:
            from ai.schemas.financial import FinancialFact

            for raw in payload.get("facts") or []:
                try:
                    state.financial_facts.append(FinancialFact.model_validate(raw))
                except Exception:  # noqa: BLE001
                    continue
