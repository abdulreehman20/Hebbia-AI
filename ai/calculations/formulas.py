"""Deterministic Decimal financial formulas."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation

from ai.schemas.financial import CalculationResult

FormulaFn = Callable[[dict[str, Decimal]], Decimal]

HUNDRED = Decimal("100")


def _require(inputs: dict[str, Decimal], *keys: str) -> None:
    missing = [key for key in keys if key not in inputs or inputs[key] is None]
    if missing:
        raise KeyError(f"Missing inputs: {missing}")


def revenue_growth(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "current_revenue", "previous_revenue")
    previous = inputs["previous_revenue"]
    if previous == 0:
        raise ZeroDivisionError("previous_revenue is zero")
    return (inputs["current_revenue"] - previous) / previous * HUNDRED


def gross_margin(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "gross_profit", "revenue")
    revenue = inputs["revenue"]
    if revenue == 0:
        raise ZeroDivisionError("revenue is zero")
    return inputs["gross_profit"] / revenue * HUNDRED


def operating_margin(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "operating_income", "revenue")
    revenue = inputs["revenue"]
    if revenue == 0:
        raise ZeroDivisionError("revenue is zero")
    return inputs["operating_income"] / revenue * HUNDRED


def net_profit_margin(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "net_income", "revenue")
    revenue = inputs["revenue"]
    if revenue == 0:
        raise ZeroDivisionError("revenue is zero")
    return inputs["net_income"] / revenue * HUNDRED


def free_cash_flow(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "operating_cash_flow", "capital_expenditure")
    return inputs["operating_cash_flow"] - inputs["capital_expenditure"]


def debt_to_equity(inputs: dict[str, Decimal]) -> Decimal:
    _require(inputs, "total_debt", "total_equity")
    equity = inputs["total_equity"]
    if equity == 0:
        raise ZeroDivisionError("total_equity is zero")
    return inputs["total_debt"] / equity


FORMULAS: dict[str, tuple[str, FormulaFn, str | None]] = {
    "revenue_growth": (
        "(Current Revenue - Previous Revenue) / Previous Revenue × 100",
        revenue_growth,
        "percent",
    ),
    "gross_margin": ("Gross Profit / Revenue × 100", gross_margin, "percent"),
    "operating_margin": ("Operating Income / Revenue × 100", operating_margin, "percent"),
    "net_profit_margin": ("Net Income / Revenue × 100", net_profit_margin, "percent"),
    "free_cash_flow": (
        "Operating Cash Flow - Capital Expenditure",
        free_cash_flow,
        "currency",
    ),
    "debt_to_equity": ("Total Debt / Total Equity", debt_to_equity, "ratio"),
}


def calculate_metric(
    metric_name: str,
    inputs: dict[str, Decimal],
    *,
    periods: dict[str, str] | None = None,
    source_chunk_ids: list[str] | None = None,
) -> CalculationResult:
    if metric_name not in FORMULAS:
        return CalculationResult(
            metric_name=metric_name,
            formula="",
            inputs=inputs,
            periods=periods or {},
            source_chunk_ids=source_chunk_ids or [],
            result=None,
            status="error",
            message=f"Unsupported metric: {metric_name}",
        )
    formula, fn, unit = FORMULAS[metric_name]
    try:
        result = fn(inputs)
        return CalculationResult(
            metric_name=metric_name,
            formula=formula,
            inputs=inputs,
            periods=periods or {},
            source_chunk_ids=source_chunk_ids or [],
            result=result,
            unit=unit,
            status="success",
        )
    except KeyError as exc:
        return CalculationResult(
            metric_name=metric_name,
            formula=formula,
            inputs=inputs,
            periods=periods or {},
            source_chunk_ids=source_chunk_ids or [],
            result=None,
            unit=unit,
            status="missing_inputs",
            message=str(exc),
        )
    except ZeroDivisionError as exc:
        return CalculationResult(
            metric_name=metric_name,
            formula=formula,
            inputs=inputs,
            periods=periods or {},
            source_chunk_ids=source_chunk_ids or [],
            result=None,
            unit=unit,
            status="invalid_inputs",
            message=str(exc),
        )
    except (InvalidOperation, DivisionByZero, ValueError) as exc:
        return CalculationResult(
            metric_name=metric_name,
            formula=formula,
            inputs=inputs,
            periods=periods or {},
            source_chunk_ids=source_chunk_ids or [],
            result=None,
            unit=unit,
            status="error",
            message=str(exc),
        )
