"""Execution limits and safety policies for the research agent."""

from __future__ import annotations

from config import Settings, get_settings


class AgentPolicies:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_iterations = self.settings.agent_max_iterations
        self.max_retrieval_calls = self.settings.max_retrieval_calls
        self.max_query_rewrites = self.settings.max_query_rewrites
        self._retrieval_calls = 0
        self._seen_tool_signatures: set[str] = set()

    def allow_iteration(self, iteration: int) -> bool:
        return iteration < self.max_iterations

    def register_retrieval(self) -> bool:
        if self._retrieval_calls >= self.max_retrieval_calls:
            return False
        self._retrieval_calls += 1
        return True

    def is_duplicate_tool_call(self, name: str, arguments: dict) -> bool:
        signature = f"{name}:{sorted(arguments.items())}"
        if signature in self._seen_tool_signatures:
            return True
        self._seen_tool_signatures.add(signature)
        return False

    def is_approved_tool(self, name: str, approved: set[str]) -> bool:
        return name in approved
