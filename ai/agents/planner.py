"""Research planning from validated intent."""

from __future__ import annotations

from ai.schemas.plan import OperationType, ResearchPlan, ResearchTask
from ai.schemas.research import ResearchIntent, ResearchRequest


class ResearchPlanner:
    """Build a validated ResearchPlan without unapproved operations."""

    def plan(self, request: ResearchRequest, intent: ResearchIntent) -> ResearchPlan:
        operations: list[OperationType] = ["retrieve_financial_evidence"]
        if intent.requires_calculation:
            operations.extend(["extract_financial_facts", "calculate_metrics"])
        if intent.requires_multiple_namespaces or intent.task_type == "company_comparison":
            operations.append("compare_companies")
        if intent.start_year and intent.end_year:
            operations.append("compare_periods")
        if intent.task_type == "risk_analysis":
            operations.append("analyze_risks")
        operations.extend(["verify_citations", "check_missing_information"])

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_ops: list[OperationType] = []
        for op in operations:
            if op not in seen:
                seen.add(op)
                unique_ops.append(op)

        tasks: list[ResearchTask] = []
        for ticker in intent.tickers:
            for query in (intent.search_queries or [request.question])[:3]:
                tasks.append(
                    ResearchTask(
                        operation="retrieve_financial_evidence",
                        tickers=[ticker],
                        query=query,
                        metrics=intent.metrics,
                        filing_types=intent.filing_types,
                        sections=intent.sections,
                    )
                )

        objective = (
            f"{intent.task_type.replace('_', ' ')} for "
            f"{', '.join(intent.tickers) or 'unspecified companies'}"
        )
        return ResearchPlan(
            task_type=intent.task_type,
            objective=objective,
            companies=intent.tickers,
            reporting_period_start=intent.start_year,
            reporting_period_end=intent.end_year,
            filing_types=intent.filing_types,
            sections=intent.sections,
            metrics=intent.metrics,
            search_queries=intent.search_queries or [request.question],
            requires_calculation=intent.requires_calculation,
            requires_comparison=intent.requires_multiple_namespaces
            or intent.task_type == "company_comparison",
            operations=unique_ops,
            tasks=tasks,
        )
