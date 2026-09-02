"""Parse the natural-language contracts emitted by the negotiation agents.

The parser is deliberately format-tolerant but semantics-strict. It accepts the
different headings and sentence layouts used by the three model providers while
keeping internal arithmetic disagreements as explicit issues. Those disagreements
are benchmark outcomes, not parser failures to hide or repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable


PRODUCTION_WEEKS = (2, 4, 6, 8, 10)
PAYMENT_WEEKS = (1, 3, 5, 7, 9, 11)
DISH_PATTERNS = {
    "A": r"(?:Margherita\s+)?Pizza(?:s)?",
    "B": r"(?:Pesto\s+)?Pasta(?:s)?",
    "C": r"(?:Tomato\s+)?Soup(?:s)?",
}


def _decimal(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def decimal_text(value: Decimal | None) -> str | None:
    """Return a stable non-scientific representation for CSV/JSON output."""
    if value is None:
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


@dataclass
class ParsedContract:
    text: str
    prices: dict[str, Decimal | None]
    deliveries: dict[int, dict[str, int]]
    payments: dict[int, Decimal]
    reported_contract_values: list[Decimal] = field(default_factory=list)
    reported_payment_totals: list[Decimal] = field(default_factory=list)
    contingencies: list[str] = field(default_factory=list)
    parse_issues: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.parse_issues

    @property
    def quantity_totals(self) -> dict[str, int] | None:
        if set(self.deliveries) != set(PRODUCTION_WEEKS):
            return None
        return {
            dish: sum(self.deliveries[week][dish] for week in PRODUCTION_WEEKS)
            for dish in DISH_PATTERNS
        }

    @property
    def scheduled_payment_total(self) -> Decimal | None:
        if set(self.payments) != set(PAYMENT_WEEKS):
            return None
        return sum(self.payments.values(), Decimal(0))

    @property
    def unit_price_total(self) -> Decimal | None:
        quantities = self.quantity_totals
        if quantities is None or any(value is None for value in self.prices.values()):
            return None
        return sum(
            (self.prices[dish] or Decimal(0)) * quantities[dish]
            for dish in DISH_PATTERNS
        )


def _dish_quantities(line: str) -> dict[str, int] | None:
    quantities: dict[str, int] = {}
    for dish, pattern in DISH_PATTERNS.items():
        match = re.search(rf"(\d+)\s+{pattern}", line, flags=re.IGNORECASE)
        if match is None:
            return None
        quantities[dish] = int(match.group(1))
    return quantities


def _price_for_dish(lines: Iterable[str], dish_pattern: str) -> Decimal | None:
    for line in lines:
        if re.search(dish_pattern, line, flags=re.IGNORECASE) is None:
            continue
        if re.search(r"delivery|production\s+(?:week|round)|payment\s+week", line, re.I):
            continue
        after = re.search(
            rf"{dish_pattern}[^\n$]*\$([0-9]+(?:\.[0-9]+)?)",
            line,
            flags=re.IGNORECASE,
        )
        before = re.search(
            rf"\$([0-9]+(?:\.[0-9]+)?)[^\n;]*?{dish_pattern}",
            line,
            flags=re.IGNORECASE,
        )
        match = after or before
        if match is not None:
            return _decimal(match.group(1))
    return None


def _explicit_deliveries(lines: Iterable[str]) -> dict[int, dict[str, int]]:
    deliveries: dict[int, dict[str, int]] = {}
    for line in lines:
        quantities = _dish_quantities(line)
        if quantities is None:
            continue
        week_matches = [
            int(match.group(1))
            for match in re.finditer(r"(?:week|wk)\s*(2|4|6|8|10)\b", line, re.I)
        ]
        if week_matches:
            deliveries[week_matches[-1]] = quantities
            continue
        round_match = re.search(r"(?:production\s+)?round\s*([1-5])\b", line, re.I)
        if round_match:
            deliveries[PRODUCTION_WEEKS[int(round_match.group(1)) - 1]] = quantities
    return deliveries


def _uniform_deliveries(text: str) -> dict[int, dict[str, int]]:
    section = re.search(
        r"(?is)deliver(?:y|ies)[^\n]*\n(.*?)(?:\npayment|\npayments)",
        text,
    )
    if section is None:
        return {}
    quantities: dict[str, int] = {}
    for dish, pattern in DISH_PATTERNS.items():
        match = re.search(rf"{pattern}\s*:\s*(\d+)", section.group(1), re.I)
        if match is None:
            return {}
        quantities[dish] = int(match.group(1))
    return {week: dict(quantities) for week in PRODUCTION_WEEKS}


def _payments(lines: Iterable[str]) -> dict[int, Decimal]:
    payments: dict[int, Decimal] = {}
    for line in lines:
        if "$" not in line or any(re.search(pattern, line, re.I) for pattern in DISH_PATTERNS.values()):
            continue
        weeks = list(re.finditer(r"week\s*(1|3|5|7|9|11)\b", line, re.I))
        amount = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", line)
        if not weeks or amount is None:
            continue
        # "Payment Week 2 (Week 3)" names the installment first and the game
        # week second; the final week occurrence is therefore authoritative.
        payments[int(weeks[-1].group(1))] = _decimal(amount.group(1))
    return payments


def _reported_values(text: str) -> tuple[list[Decimal], list[Decimal]]:
    contract_values: list[Decimal] = []
    payment_totals: list[Decimal] = []
    for raw in text.splitlines():
        line = raw.strip().strip("*- ")
        amounts = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", line)
        if re.match(
            r"(?:contract value|contract total|total contract price|total contract payment)",
            line,
            re.I,
        ) and amounts:
            # Formula lines contain several currency amounts. The last one is
            # the stated result after the final '='.
            contract_values.append(_decimal(amounts[-1]))
            continue
        if re.match(
            r"total(?:\s+(?:paid|payment|payments|contract payment|contract price))?\s*:",
            line,
            re.I,
        ) and amounts:
            payment_totals.append(_decimal(amounts[-1]))
    return contract_values, payment_totals


def _contingencies(text: str) -> list[str]:
    lower = text.lower()
    patterns = {
        "substitution": ("substitution", "substitute", "replacement"),
        "payment_deduction": ("deduct", "deduction", "credit against", "pro rata"),
        "rollover": ("rollover", "carry forward", "carry-forward", "made up in"),
        "grim_trigger": ("terminate", "termination", "walk away", "contract ends"),
    }
    return [name for name, phrases in patterns.items() if any(phrase in lower for phrase in phrases)]


def parse_contract(text: str) -> ParsedContract:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prices = {
        dish: _price_for_dish(lines, pattern)
        for dish, pattern in DISH_PATTERNS.items()
    }
    deliveries = _explicit_deliveries(lines)
    if set(deliveries) != set(PRODUCTION_WEEKS):
        uniform = _uniform_deliveries(text)
        if uniform:
            deliveries = uniform
    payments = _payments(lines)
    contract_values, payment_totals = _reported_values(text)

    issues: list[str] = []
    for dish, price in prices.items():
        if price is None:
            issues.append(f"missing_price:{dish}")
    if set(deliveries) != set(PRODUCTION_WEEKS):
        issues.append("delivery_weeks:" + ",".join(map(str, sorted(deliveries))))
    if set(payments) != set(PAYMENT_WEEKS):
        issues.append("payment_weeks:" + ",".join(map(str, sorted(payments))))

    return ParsedContract(
        text=text,
        prices=prices,
        deliveries=deliveries,
        payments=payments,
        reported_contract_values=contract_values,
        reported_payment_totals=payment_totals,
        contingencies=_contingencies(text),
        parse_issues=issues,
    )
