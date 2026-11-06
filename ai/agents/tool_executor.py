"""Validate and execute registered LangChain tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool

from ai.agents.policies import AgentPolicies
from ai.agents.state import ToolCallRecord
from ai.tools.registry import get_tool_map


class ToolExecutor:
    def __init__(self, policies: AgentPolicies | None = None) -> None:
        self.policies = policies or AgentPolicies()
        self.tools: dict[str, BaseTool] = get_tool_map()
        self.approved = set(self.tools)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        if not self.policies.is_approved_tool(name, self.approved):
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result_preview=f"Rejected unapproved tool: {name}",
                success=False,
            )
        if self.policies.is_duplicate_tool_call(name, arguments):
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result_preview="Skipped duplicate tool call",
                success=False,
            )
        if name.startswith("search_") or name in {
            "expand_document_context",
            "find_related_amendments",
            "analyze_risk_changes",
        }:
            if not self.policies.register_retrieval():
                return ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    result_preview="Retrieval call budget exceeded",
                    success=False,
                )
        tool = self.tools[name]
        try:
            result = tool.invoke(arguments)
            preview = result if isinstance(result, str) else json.dumps(result, default=str)[:1500]
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result_preview=preview,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                result_preview=f"Tool error: {exc}",
                success=False,
            )
