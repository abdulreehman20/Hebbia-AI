"""Typed agent state for the Financial Research Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.schemas.answers import ResearchAnswer
from ai.schemas.evidence import MissingEvidence
from ai.schemas.financial import CalculationResult, FinancialFact
from ai.schemas.plan import ResearchPlan
from ai.schemas.research import ResearchIntent, ResearchRequest
from ai.schemas.retrieval import RetrievedChunk
from ai.schemas.citations import SourceCitation

ExecutionStatus = Literal[
    "started",
    "planning",
    "retrieving",
    "analyzing",
    "generating",
    "completed",
    "failed",
    "unsupported",
]


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str | None = None
    success: bool = True


class ResearchState(BaseModel):
    request_id: str
    request: ResearchRequest
    intent: ResearchIntent | None = None
    plan: ResearchPlan | None = None
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    financial_facts: list[FinancialFact] = Field(default_factory=list)
    calculations: list[CalculationResult] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    missing: list[MissingEvidence] = Field(default_factory=list)
    status: ExecutionStatus = "started"
    error: str | None = None
    answer: ResearchAnswer | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)

    def emit(self, event_type: str, **payload: Any) -> None:
        self.events.append({"event": event_type, **payload})
