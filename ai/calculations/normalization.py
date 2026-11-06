"""Normalize financial scales and currencies before calculation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

SCALE_MULTIPLIERS: dict[str, Decimal] = {
    "ones": Decimal("1"),
    "thousands": Decimal("1000"),
    "millions": Decimal("1000000"),
    "billions": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "trillions": Decimal("1000000000000"),
}


def parse_decimal(value: str | int | float | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot parse decimal from {value!r}") from exc


def normalize_to_ones(value: Decimal, scale: str | None) -> Decimal:
    if not scale:
        return value
    key = scale.strip().lower()
    multiplier = SCALE_MULTIPLIERS.get(key)
    if multiplier is None:
        return value
    return value * multiplier


def scales_compatible(scale_a: str | None, scale_b: str | None) -> bool:
    if not scale_a or not scale_b:
        return True
    return scale_a.strip().lower() == scale_b.strip().lower()


def currencies_compatible(currency_a: str | None, currency_b: str | None) -> bool:
    if not currency_a or not currency_b:
        return True
    return currency_a.strip().upper() == currency_b.strip().upper()
