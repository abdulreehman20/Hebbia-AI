"""Integration tests for research API with mocked agent."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from ai.agents.state import ResearchState
from ai.schemas.answers import ResearchAnswer
from ai.schemas.research import ResearchRequest
from main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_stream_rejects_bad_ticker() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/research/stream",
        json={"question": "Tesla revenue?", "selected_companies": ["TSLA"]},
    )
    assert response.status_code == 422


def test_research_stream_sse_with_mocked_agent() -> None:
    request = ResearchRequest(question="What is Apple revenue?", selected_companies=["AAPL"])
    state = ResearchState(request_id="test-1", request=request, status="completed")
    state.emit("research_started", request_id="test-1")
    state.emit("planning_completed", task_type="revenue_and_profitability")
    state.answer = ResearchAnswer(
        executive_summary="Insufficient live corpus for a full answer.",
        insufficient_evidence=True,
        missing_information=["Pinecone ingest incomplete"],
    )

    def fake_stream(self, req: ResearchRequest):  # noqa: ANN001
        yield from state.events
        yield {
            "event": "answer_completed",
            "request_id": state.request_id,
            "status": state.status,
            "answer": state.answer.model_dump(mode="json"),
            "error": None,
        }

    with patch("ai.api.research.FinancialResearchAgent.stream_events", fake_stream):
        client = TestClient(app)
        with client.stream(
            "POST",
            "/api/v1/research/stream",
            json={"question": "What is Apple revenue?", "selected_companies": ["AAPL"]},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
    assert "research_started" in body
    assert "answer_completed" in body
