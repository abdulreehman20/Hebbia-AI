"""Approved internal LangChain tool registry."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ai.tools.evidence_tools import (
    check_missing_information,
    verify_citations,
    verify_financial_calculations,
    verify_retrieved_evidence,
)
from ai.tools.filing_tools import expand_document_context, find_related_amendments
from ai.tools.financial_tools import (
    analyze_risk_changes,
    build_financial_summary,
    calculate_financial_metrics,
    compare_companies,
    compare_financial_periods,
    extract_financial_facts,
)
from ai.tools.search_tools import (
    filter_filings,
    search_company_filings,
    search_filing_section,
    search_multiple_companies,
)


def get_research_tools() -> list[BaseTool]:
    """Return only explicitly approved internal research tools."""
    return [
        search_company_filings,
        search_multiple_companies,
        filter_filings,
        search_filing_section,
        expand_document_context,
        find_related_amendments,
        extract_financial_facts,
        calculate_financial_metrics,
        compare_financial_periods,
        compare_companies,
        analyze_risk_changes,
        build_financial_summary,
        verify_retrieved_evidence,
        verify_financial_calculations,
        verify_citations,
        check_missing_information,
    ]


def get_tool_map() -> dict[str, BaseTool]:
    return {tool.name: tool for tool in get_research_tools()}
