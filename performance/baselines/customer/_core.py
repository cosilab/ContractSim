"""Shared customer-side logic for the RC, RE, and RCC baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from performance.engine.environment import DISHES, load_performance_environment


def _dish_zero() -> dict[str, int]:
    return {dish: 0 for dish in DISHES}


def _int_dishes(row: dict[str, Any]) -> dict[str, int]:
    return {dish: int(row.get(dish, 0)) for dish in DISHES}


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {dish: int(a.get(dish, 0)) + int(b.get(dish, 0)) for dish in DISHES}


def _sub_nonnegative(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {dish: max(0, int(a.get(dish, 0)) - int(b.get(dish, 0))) for dish in DISHES}


def _shortfall_value(shortfall: dict[str, int], prices: dict[str, int]) -> float:
    return float(sum(int(shortfall[dish]) * float(prices[dish]) for dish in DISHES))


def _soup_value(
    qty: dict[str, int],
    soup_equivalents: dict[str, int],
) -> int:
    return int(
        sum(
            int(qty[dish]) * int(soup_equivalents[dish])
            for dish in DISHES
        )
    )


def _min_qty_from_config(
    config: dict[str, Any],
    default_min_qty: dict[str, int],
) -> dict[str, int]:
    params = config.get("contingency_params", {}) or {}
    for name in ("substitution", "payment_deduction"):
        min_qty = (params.get(name) or {}).get("min_qty")
        if min_qty:
            return {
                dish: int(min_qty.get(dish, default_min_qty[dish]))
                for dish in DISHES
            }
    return dict(default_min_qty)


def _rollover_max_deficit(config: dict[str, Any]) -> int:
    params = config.get("contingency_params", {}) or {}
    raw = (params.get("rollover") or {}).get("max_deficit")
    return 2 if raw is None else int(raw)


def _deduction_prices_from_config(
    config: dict[str, Any],
    fallback: dict[str, int],
) -> dict[str, int]:
    params = config.get("contingency_params", {}) or {}
    raw = (params.get("payment_deduction") or {}).get("deduction_prices")
    if not raw:
        raw = fallback
    return {dish: int(raw.get(dish, fallback[dish])) for dish in DISHES}


def payment_schedule_from_terms(terms: dict[str, Any]) -> dict[int, float]:
    return {
        int(row["week"]): float(row["amount"])
        for row in terms.get("payment_schedule", [])
    }


def production_schedule_from_terms(terms: dict[str, Any]) -> dict[int, dict[str, int]]:
    return {
        int(row["week"]): _int_dishes(row)
        for row in terms.get("production_schedule", [])
    }


def _shortfall_after_substitution(
    target: dict[str, int],
    produced: dict[str, int],
    *,
    exact_req: dict[str, int],
    allow_substitution: bool,
    weights: dict[str, int],
    soup_equivalents: dict[str, int],
) -> tuple[dict[str, int], bool]:
    """Return residual shortfall and whether exact named requirements were met.

    Substitution is only allowed above exact named requirements. In rollover
    contracts, carried deficits are exact named requirements too; this mirrors
    the solver's `_rollover_exact_schedule` behavior.
    """
    target = {dish: int(target.get(dish, 0)) for dish in DISHES}
    produced = {dish: int(produced.get(dish, 0)) for dish in DISHES}
    exact_req = {
        dish: min(int(target[dish]), int(exact_req.get(dish, 0)))
        for dish in DISHES
    }

    exact_served = {
        dish: min(produced[dish], exact_req[dish])
        for dish in DISHES
    }
    exact_met = all(exact_served[dish] == exact_req[dish] for dish in DISHES)

    if not allow_substitution:
        return _sub_nonnegative(target, produced), exact_met

    target_above = {
        dish: max(0, target[dish] - exact_req[dish])
        for dish in DISHES
    }
    produced_above = {
        dish: max(0, produced[dish] - exact_req[dish])
        for dish in DISHES
    }
    budget = _soup_value(produced_above, soup_equivalents)

    best_credited = _dish_zero()
    best_score: tuple[float, int, int] | None = None
    for a in range(target_above["A"] + 1):
        for b in range(target_above["B"] + 1):
            for c in range(target_above["C"] + 1):
                credited = {"A": a, "B": b, "C": c}
                soup = _soup_value(credited, soup_equivalents)
                if soup > budget:
                    continue
                price_credit = _shortfall_value(credited, weights)
                units = a + b + c
                score = (price_credit, soup, units)
                if best_score is None or score > best_score:
                    best_score = score
                    best_credited = credited

    served = _add(exact_served, best_credited)
    return _sub_nonnegative(target, served), exact_met


@dataclass
class CustomerBaseline:
    """Stateful payment policy for the customer side."""

    baseline: str
    config: dict[str, Any]
    terms: dict[str, Any]
    private_values: dict[str, int] | None = None
    soup_equivalents: dict[str, int] | None = None
    default_min_qty: dict[str, int] | None = None
    environment_id: int = 1
    payment_schedule: dict[int, float] = field(init=False)
    production_schedule: dict[int, dict[str, int]] = field(init=False)
    dish_prices: dict[str, int] = field(init=False)
    deduction_prices: dict[str, int] = field(init=False)
    contingencies: set[str] = field(init=False)
    min_qty: dict[str, int] = field(init=False)
    rollover_max_deficit: int = field(init=False)
    carry_deficit: dict[str, int] = field(default_factory=_dish_zero)
    suspended: bool = False
    had_prior_shortfall: bool = False
    pending_deduction: float = 0.0
    total_paid: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.baseline not in {"rc", "rcc", "re"}:
            raise ValueError(f"Unknown customer baseline: {self.baseline}")
        environment = load_performance_environment(self.environment_id)
        self.private_values = {
            dish: int(
                (self.private_values or environment.customer_private_values)[dish]
            )
            for dish in DISHES
        }
        self.soup_equivalents = {
            dish: int(
                (self.soup_equivalents or environment.soup_equivalents)[dish]
            )
            for dish in DISHES
        }
        self.default_min_qty = {
            dish: int(
                (
                    self.default_min_qty
                    or environment.customer_minimum_quantities
                )[dish]
            )
            for dish in DISHES
        }
        self.payment_schedule = payment_schedule_from_terms(self.terms)
        self.production_schedule = production_schedule_from_terms(self.terms)
        self.dish_prices = {
            dish: int(self.terms["dish_prices"][dish])
            for dish in DISHES
        }
        self.deduction_prices = _deduction_prices_from_config(
            self.config,
            self.dish_prices,
        )
        self.contingencies = set(self.config.get("contingencies", []) or [])
        self.min_qty = _min_qty_from_config(self.config, self.default_min_qty)
        self.rollover_max_deficit = _rollover_max_deficit(self.config)

    @property
    def allow_substitution(self) -> bool:
        return "substitution" in self.contingencies

    @property
    def allow_rollover(self) -> bool:
        return "rollover" in self.contingencies

    @property
    def allow_payment_deduction(self) -> bool:
        return "payment_deduction" in self.contingencies

    @property
    def baseline_type(self) -> str:
        return {
            "rc": "rational_complier_customer",
            "rcc": "rational_conditional_complier_customer",
            "re": "rational_exploiter_customer",
        }[self.baseline]

    def scheduled_payment(self, week: int) -> float:
        return float(self.payment_schedule.get(int(week), 0.0))

    def decide_payment(self, *, week: int, remaining_budget: float) -> tuple[float, dict[str, Any]]:
        scheduled = self.scheduled_payment(week)
        reason = "scheduled_payment"

        if self.baseline == "re":
            payment = 0.0
            reason = "re_never_pays"
        elif self.baseline == "rcc" and self.suspended:
            payment = 0.0
            reason = "rcc_suspended_after_uncured_default"
        else:
            deduction = min(self.pending_deduction, scheduled)
            payment = scheduled - deduction
            if deduction > 0:
                reason = f"{self.baseline}_payment_deduction"
            else:
                reason = f"{self.baseline}_pays_legal_due"
            self.pending_deduction = 0.0

        payment = round(max(0.0, min(float(payment), float(remaining_budget))), 2)
        self.total_paid += payment
        decision = {
            "week": int(week),
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "scheduled_payment": scheduled,
            "payment": payment,
            "reason": reason,
            "suspended": self.suspended,
            "carry_deficit": dict(self.carry_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
        }
        self.history.append({"event": "payment", **decision})
        return payment, decision

    def observe_production(self, *, week: int, produced: dict[str, int]) -> dict[str, Any]:
        scheduled_target = self.production_schedule.get(int(week), _dish_zero())
        effective_target = (
            _add(scheduled_target, self.carry_deficit)
            if self.allow_rollover
            else dict(scheduled_target)
        )
        had_carry = any(self.carry_deficit[dish] > 0 for dish in DISHES)
        exact_req = {
            dish: min(scheduled_target[dish], self.min_qty[dish])
            for dish in DISHES
        }
        if self.allow_rollover:
            exact_req = _add(exact_req, self.carry_deficit)
        substitution_weights = (
            self.deduction_prices
            if self.allow_payment_deduction
            else self.private_values
        )

        shortfall, minima_met = _shortfall_after_substitution(
            effective_target,
            produced,
            exact_req=exact_req,
            allow_substitution=self.allow_substitution,
            weights=substitution_weights,
            soup_equivalents=self.soup_equivalents,
        )
        if (
            self.allow_substitution
            and (self.allow_payment_deduction or self.allow_rollover)
            and any(shortfall.values())
        ):
            shortfall = _sub_nonnegative(effective_target, produced)
        shortfall_value = _shortfall_value(
            shortfall,
            self.deduction_prices if self.allow_payment_deduction else self.dish_prices,
        )

        breach = any(shortfall[dish] > 0 for dish in DISHES)
        is_final_round = int(week) == max(self.production_schedule, default=-1)
        rollover_limit_breached = any(
            shortfall[dish] > self.rollover_max_deficit
            for dish in DISHES
        )
        uncured_carry = had_carry and breach

        if self.baseline in {"rc", "rcc"}:
            if self.allow_rollover:
                if self.baseline == "rcc" and (
                    rollover_limit_breached
                    or uncured_carry
                    or (is_final_round and breach)
                ):
                    self.suspended = True
                self.carry_deficit = dict(shortfall)
            elif self.allow_payment_deduction:
                if self.baseline == "rcc" and self.had_prior_shortfall and breach:
                    self.suspended = True
                self.carry_deficit = _dish_zero()
                self.had_prior_shortfall = breach
            elif breach:
                if self.baseline == "rcc":
                    self.suspended = True
                self.carry_deficit = _dish_zero()
            else:
                self.carry_deficit = _dish_zero()
                self.had_prior_shortfall = False

            if self.allow_payment_deduction:
                self.pending_deduction = shortfall_value if breach else 0.0

        record = {
            "event": "production_observed",
            "week": int(week),
            "baseline": self.baseline,
            "scheduled_target": scheduled_target,
            "effective_target": effective_target,
            "produced": {dish: int(produced.get(dish, 0)) for dish in DISHES},
            "shortfall": dict(shortfall),
            "shortfall_value": round(shortfall_value, 2),
            "minima_met": bool(minima_met),
            "breach": bool(breach),
            "had_carry": bool(had_carry),
            "rollover_limit_breached": bool(rollover_limit_breached),
            "uncured_carry": bool(uncured_carry),
            "suspended": self.suspended,
            "carry_deficit": dict(self.carry_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
        }
        self.history.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "environment_id": self.environment_id,
            "total_paid": round(self.total_paid, 2),
            "suspended": self.suspended,
            "carry_deficit": dict(self.carry_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
            "history": self.history,
        }
