"""Validated, immutable contract terms for the RC solver.

This first contract model intentionally covers only the core exchange:
scheduled product quantities, product prices, and payments. Contingency terms
will be added as a separate concern later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from environment.json_input import (
    immutable_integer_mapping,
    load_json_object,
    require_finite_number,
    require_integer,
    require_json_mapping,
)

@dataclass(frozen=True)
class ProductionWeek:
    """Product quantities due in one production week."""

    week: int
    product_quantities: Mapping[str, int]


@dataclass(frozen=True)
class PaymentWeek:
    """Payment due in one payment week."""

    week: int
    amount: float


def _required_list(source: Mapping[str, Any], key: str) -> list[Any]:
    value = source.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"contract.{key} must be a non-empty list")
    return value


def _immutable_price_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")

    normalized: dict[str, float] = {}
    for product, raw_price in value.items():
        if not isinstance(product, str) or not product:
            raise ValueError(f"{name} keys must be non-empty product names")
        normalized[product] = require_finite_number(
            raw_price,
            name=f"{name}[{product!r}]",
            minimum=0,
        )
    if not normalized:
        raise ValueError(f"{name} must contain at least one product")
    return MappingProxyType(normalized)


def _validate_schedule_weeks(
    actual: tuple[int, ...],
    *,
    schedule_name: str,
    parity_name: str,
    parity: int,
) -> None:
    if len(actual) != len(set(actual)):
        raise ValueError(f"{schedule_name} cannot contain duplicate weeks")
    invalid = sorted(week for week in actual if week % 2 != parity)
    if invalid:
        raise ValueError(
            f"{schedule_name} weeks must be {parity_name}; got {invalid}"
        )


def _validate_combined_calendar(
    production_schedule: tuple[ProductionWeek, ...],
    payment_schedule: tuple[PaymentWeek, ...],
) -> None:
    if len(payment_schedule) != len(production_schedule) + 1:
        raise ValueError(
            "The payment schedule must contain one payment before each "
            "production week and one final payment"
        )

    calendar = tuple(sorted(
        [entry.week for entry in production_schedule]
        + [entry.week for entry in payment_schedule]
    ))
    expected_calendar: list[int] = []
    for payment, production in zip(payment_schedule, production_schedule):
        expected_calendar.extend((payment.week, production.week))
    expected_calendar.append(payment_schedule[-1].week)
    if calendar != tuple(expected_calendar):
        raise ValueError(
            "The contract calendar must begin with a payment, alternate "
            "payment and production weeks, and end with a final payment"
        )

    step = calendar[1] - calendar[0]
    if any(
        current - previous != step
        for previous, current in zip(calendar, calendar[1:])
    ):
        raise ValueError(
            "Combined payment and production weeks must form an arithmetic progression"
        )


def _parse_production_schedule(
    source: Mapping[str, Any],
) -> tuple[tuple[ProductionWeek, ...], tuple[str, ...]]:
    raw_schedule = _required_list(source, "production_schedule")
    entries: list[ProductionWeek] = []
    products: tuple[str, ...] | None = None

    for index, raw_entry in enumerate(raw_schedule):
        path = f"contract.production_schedule[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path} must be a mapping")

        week = require_integer(raw_entry.get("week"), name=f"{path}.week", minimum=1)
        quantities = immutable_integer_mapping(
            raw_entry.get("product_quantities"),
            name=f"{path}.product_quantities",
            minimum=0,
        )

        entry_products = tuple(quantities)
        if products is None:
            products = entry_products
        elif set(entry_products) != set(products):
            raise ValueError(
                "Every production week must specify the same set of products"
            )

        entries.append(
            ProductionWeek(
                week=week,
                product_quantities=quantities,
            )
        )

    weeks = tuple(entry.week for entry in entries)
    _validate_schedule_weeks(
        weeks,
        schedule_name="production_schedule",
        parity_name="even",
        parity=0,
    )
    entries.sort(key=lambda entry: entry.week)
    assert products is not None
    return tuple(entries), products


def _parse_payment_schedule(
    source: Mapping[str, Any],
) -> tuple[PaymentWeek, ...]:
    raw_schedule = _required_list(source, "payment_schedule")
    entries: list[PaymentWeek] = []

    for index, raw_entry in enumerate(raw_schedule):
        path = f"contract.payment_schedule[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path} must be a mapping")
        entries.append(
            PaymentWeek(
                week=require_integer(
                    raw_entry.get("week"),
                    name=f"{path}.week",
                    minimum=1,
                ),
                amount=require_finite_number(
                    raw_entry.get("amount"),
                    name=f"{path}.amount",
                    minimum=0,
                ),
            )
        )

    weeks = tuple(entry.week for entry in entries)
    _validate_schedule_weeks(
        weeks,
        schedule_name="payment_schedule",
        parity_name="odd",
        parity=1,
    )
    entries.sort(key=lambda entry: entry.week)
    return tuple(entries)


@dataclass(frozen=True, init=False)
class Contract:
    """Core contract terms built only from a JSON-shaped mapping."""

    products: tuple[str, ...]
    product_prices: Mapping[str, float]
    production_schedule: tuple[ProductionWeek, ...]
    payment_schedule: tuple[PaymentWeek, ...]

    def __init__(self, source: Mapping[str, Any]) -> None:
        source = require_json_mapping(source, name="Contract input")
        production_schedule, products = _parse_production_schedule(source)
        product_prices = _immutable_price_mapping(
            source.get("product_prices"),
            name="contract.product_prices",
        )
        if set(product_prices) != set(products):
            raise ValueError(
                "contract.product_prices products must match production products"
            )
        payment_schedule = _parse_payment_schedule(source)
        _validate_combined_calendar(production_schedule, payment_schedule)

        object.__setattr__(self, "products", products)
        object.__setattr__(self, "product_prices", product_prices)
        object.__setattr__(self, "production_schedule", production_schedule)
        object.__setattr__(self, "payment_schedule", payment_schedule)

    @property
    def production_weeks(self) -> tuple[int, ...]:
        return tuple(entry.week for entry in self.production_schedule)

    @property
    def payment_weeks(self) -> tuple[int, ...]:
        return tuple(entry.week for entry in self.payment_schedule)

    @property
    def calendar_weeks(self) -> tuple[int, ...]:
        """Return all scheduled contract weeks in chronological order."""
        return tuple(sorted(self.production_weeks + self.payment_weeks))

    @property
    def number_of_weeks(self) -> int:
        """Return the number of weeks inferred from the combined calendar."""
        return len(self.calendar_weeks)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "Contract":
        """Build from an already parsed contract JSON object."""
        return cls(source)

    @classmethod
    def from_json(cls, path: str | Path) -> "Contract":
        """Load a contract from a JSON file."""
        return cls(load_json_object(path, name="Contract JSON"))
