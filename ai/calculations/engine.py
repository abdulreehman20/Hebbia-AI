"""Calculation engine wrapping formulas and fact inputs."""

from __future__ import annotations

from decimal import Decimal

from ai.calculations.formulas import FORMULAS, calculate_metric
from ai.calculations.normalization import (
    currencies_compatible,
    normalize_to_ones,
    parse_decimal,
    scales_compatible,
)
from ai.schemas.financial import CalculationResult, FinancialFact


class CalculationEngine:
    """Run deterministic calculations from validated FinancialFact inputs."""

    supported_metrics = frozenset(FORMULAS)

    def calculate(
        self,
        metric_name: str,
        inputs: dict[str, Decimal | str | int | float],
        *,
        periods: dict[str, str] | None = None,
        source_chunk_ids: list[str] | None = None,
    ) -> CalculationResult:
        parsed = {key: parse_decimal(value) for key, value in inputs.items()}
        return calculate_metric(
            metric_name,
            parsed,
            periods=periods,
            source_chunk_ids=source_chunk_ids,
        )

    def calculate_from_facts(
        self,
        metric_name: str,
        mapping: dict[str, FinancialFact],
        *,
        periods: dict[str, str] | None = None,
    ) -> CalculationResult:
        facts = list(mapping.values())
        if len(facts) >= 2:
            for left, right in zip(facts, facts[1:], strict=False):
                if not currencies_compatible(left.currency, right.currency):
                    return CalculationResult(
                        metric_name=metric_name,
                        formula=FORMULAS.get(metric_name, ("", None, None))[0],
                        inputs={},
                        periods=periods or {},
                        source_chunk_ids=[f.chunk_id for f in facts],
                        result=None,
                        status="invalid_inputs",
                        message="Currency mismatch across inputs",
                    )
                if not scales_compatible(left.scale, right.scale):
                    # Normalize both to ones when scales differ but are known.
                    pass

        inputs: dict[str, Decimal] = {}
        chunk_ids: list[str] = []
        for key, fact in mapping.items():
            inputs[key] = normalize_to_ones(fact.value, fact.scale)
            chunk_ids.append(fact.chunk_id)
        return calculate_metric(
            metric_name,
            inputs,
            periods=periods,
            source_chunk_ids=chunk_ids,
        )
