"""Shared supplier-side logic for the RC, RE, and RCC baselines."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import product as iproduct
from pathlib import Path
from typing import Any

import numpy as np

from performance.engine.environment import load_performance_environment

DISHES = ("A", "B", "C")
INGS = ("X", "Y", "Z")
RECIPES = {
    "A": {"X": 1, "Y": 1, "Z": 1},
    "B": {"X": 1, "Y": 1},
    "C": {"Z": 1},
}
SOUP_EQ = {"A": 4, "B": 2, "C": 1}
PRIVATE_VALUE = {"A": 12, "B": 7, "C": 5}
DEFAULT_MIN_QTY = {"A": 1, "B": 2, "C": 1}

CONTINGENCY_GRIM = "grim_trigger"
CONTINGENCY_PAYDED = "payment_deduction"
CONTINGENCY_ROLLOVER = "rollover"
CONTINGENCY_PAYDED_ROLLOVER = "payment_deduction_rollover"
CONTINGENCY_SUBSTITUTION = "substitution"

POLICY_CACHE_SIZE = 1
_POLICY_CACHE: OrderedDict[Path, np.ndarray] = OrderedDict()


def _dish_zero() -> dict[str, int]:
    return {dish: 0 for dish in DISHES}


def _ing_zero() -> dict[str, int]:
    return {ing: 0 for ing in INGS}


def _int_dishes(row: dict[str, Any]) -> dict[str, int]:
    return {dish: int(row.get(dish, 0)) for dish in DISHES}


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {dish: int(a.get(dish, 0)) + int(b.get(dish, 0)) for dish in DISHES}


def _sub_nonnegative(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {dish: max(0, int(a.get(dish, 0)) - int(b.get(dish, 0))) for dish in DISHES}


def _soup_value(
    qty: dict[str, int],
    soup_equivalents: dict[str, int] = SOUP_EQ,
) -> int:
    return int(
        sum(
            int(qty[dish]) * int(soup_equivalents[dish])
            for dish in DISHES
        )
    )


def _weighted_value(qty: dict[str, int], weights: dict[str, int]) -> int:
    return int(sum(int(qty[dish]) * int(weights[dish]) for dish in DISHES))


def _usage_for(prod: dict[str, int]) -> dict[str, int]:
    usage = {ing: 0 for ing in INGS}
    for dish in DISHES:
        for ing, qty in RECIPES[dish].items():
            usage[ing] += int(prod[dish]) * int(qty)
    return usage


def _can_make(prod: dict[str, int], inventory: dict[str, int]) -> bool:
    usage = _usage_for(prod)
    return all(usage[ing] <= int(inventory.get(ing, 0)) for ing in INGS)


def _payment_schedule_from_terms(terms: dict[str, Any]) -> dict[int, float]:
    return {
        int(row["week"]): float(row["amount"])
        for row in terms.get("payment_schedule", [])
    }


def _production_schedule_from_terms(terms: dict[str, Any]) -> dict[int, dict[str, int]]:
    return {
        int(row["week"]): _int_dishes(row)
        for row in terms.get("production_schedule", [])
    }


def _normalize_round_schedule(
    raw: Any,
    *,
    n_rounds: int,
    default_row: dict[str, int],
) -> list[dict[str, int]]:
    if raw is None:
        rows = [default_row] * n_rounds
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = [raw] * n_rounds
    if len(rows) != n_rounds:
        raise ValueError(f"Expected {n_rounds} schedule rows, got {len(rows)}")
    return [
        {dish: int((row or default_row).get(dish, default_row[dish])) for dish in DISHES}
        for row in rows
    ]


def _contingency_params(config: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((config.get("contingency_params") or {}).get(name) or {})


def _min_qty_schedule(
    config: dict[str, Any],
    production_weeks: list[int],
    default_min_qty: dict[str, int] = DEFAULT_MIN_QTY,
) -> list[dict[str, int]]:
    params = _contingency_params(config, CONTINGENCY_SUBSTITUTION)
    raw = params.get("min_qty_schedule", params.get("min_qty"))
    return _normalize_round_schedule(
        raw,
        n_rounds=len(production_weeks),
        default_row=default_min_qty,
    )


def _deduction_price_schedule(
    config: dict[str, Any],
    production_weeks: list[int],
    private_values: dict[str, int] = PRIVATE_VALUE,
) -> list[dict[str, int]]:
    params = _contingency_params(config, CONTINGENCY_PAYDED)
    raw = params.get("deduction_prices_schedule", params.get("deduction_prices"))
    return _normalize_round_schedule(
        raw,
        n_rounds=len(production_weeks),
        default_row={
            dish: int(
                config.get("dish_prices", {}).get(dish, private_values[dish])
            )
            for dish in DISHES
        },
    )


def _rollover_max_deficit(config: dict[str, Any]) -> int:
    params = _contingency_params(config, CONTINGENCY_ROLLOVER)
    raw = params.get("max_deficit")
    return 2 if raw is None else int(raw)


def _candidate_upper_bounds(
    target: dict[str, int],
    exact_req: dict[str, int],
    allow_substitution: bool,
    soup_equivalents: dict[str, int] = SOUP_EQ,
) -> dict[str, int]:
    if not allow_substitution:
        return dict(target)
    extra_target_eq = sum(
        max(int(target[dish]) - int(exact_req[dish]), 0)
        * soup_equivalents[dish]
        for dish in DISHES
    )
    return {
        dish: max(
            int(target[dish]),
            int(exact_req[dish])
            + math.ceil(extra_target_eq / soup_equivalents[dish]),
        )
        for dish in DISHES
    }


def _best_extra_service(
    extra_target: dict[str, int],
    capacity_eq: int,
    weights: dict[str, int],
    soup_equivalents: dict[str, int] = SOUP_EQ,
) -> dict[str, int]:
    best = _dish_zero()
    best_score = (-1, -1, -1)
    for a in range(int(extra_target["A"]) + 1):
        for b in range(int(extra_target["B"]) + 1):
            for c in range(int(extra_target["C"]) + 1):
                served = {"A": a, "B": b, "C": c}
                eq = _soup_value(served, soup_equivalents)
                if eq > capacity_eq:
                    continue
                score = (_weighted_value(served, weights), eq, a + b + c)
                if score > best_score:
                    best = served
                    best_score = score
    return best


def _delivery_profile(
    produced: dict[str, int],
    target: dict[str, int],
    exact_req: dict[str, int],
    *,
    allow_substitution: bool,
    require_complete_substitution: bool = False,
    weights: dict[str, int],
    soup_equivalents: dict[str, int] = SOUP_EQ,
    private_values: dict[str, int] = PRIVATE_VALUE,
) -> dict[str, Any]:
    exact_served = {
        dish: min(int(produced[dish]), int(exact_req[dish]))
        for dish in DISHES
    }
    extra_target = {
        dish: max(int(target[dish]) - int(exact_req[dish]), 0)
        for dish in DISHES
    }
    extra_produced = {
        dish: max(int(produced[dish]) - int(exact_req[dish]), 0)
        for dish in DISHES
    }
    if allow_substitution:
        capacity_eq = _soup_value(extra_produced, soup_equivalents)
        extra_served = _best_extra_service(
            extra_target,
            capacity_eq,
            weights,
            soup_equivalents,
        )
    else:
        extra_served = {
            dish: min(extra_produced[dish], extra_target[dish])
            for dish in DISHES
        }
    served = _add(exact_served, extra_served)
    shortfall = _sub_nonnegative(target, served)
    if (
        allow_substitution
        and require_complete_substitution
        and any(shortfall.values())
    ):
        served = {
            dish: min(int(produced[dish]), int(target[dish]))
            for dish in DISHES
        }
        shortfall = _sub_nonnegative(target, produced)
    served_weight = sum(
        (int(target[dish]) - int(shortfall[dish])) * int(weights[dish])
        for dish in DISHES
    )
    return {
        "exact_served": exact_served,
        "served": served,
        "shortfall": shortfall,
        "served_weight": int(served_weight),
        "shortfall_value": _weighted_value(shortfall, weights),
        "full_satisfied": all(shortfall[dish] == 0 for dish in DISHES),
        "produced_soup": _soup_value(produced, soup_equivalents),
        "produced_value": _weighted_value(produced, private_values),
        "total_usage": sum(_usage_for(produced).values()),
    }


def _production_candidates(
    target: dict[str, int],
    exact_req: dict[str, int],
    *,
    allow_substitution: bool,
    require_complete_substitution: bool,
    weights: dict[str, int],
    inventory: dict[str, int],
    soup_equivalents: dict[str, int] = SOUP_EQ,
    private_values: dict[str, int] = PRIVATE_VALUE,
) -> list[tuple[dict[str, int], dict[str, Any]]]:
    upper = _candidate_upper_bounds(
        target,
        exact_req,
        allow_substitution,
        soup_equivalents,
    )
    candidates: list[tuple[dict[str, int], dict[str, Any]]] = []
    for a, b, c in iproduct(
        range(int(upper["A"]) + 1),
        range(int(upper["B"]) + 1),
        range(int(upper["C"]) + 1),
    ):
        produced = {"A": a, "B": b, "C": c}
        if not _can_make(produced, inventory):
            continue
        profile = _delivery_profile(
            produced,
            target,
            exact_req,
            allow_substitution=allow_substitution,
            require_complete_substitution=require_complete_substitution,
            weights=weights,
            soup_equivalents=soup_equivalents,
            private_values=private_values,
        )
        candidates.append((produced, profile))
    return candidates


def _coerce_scalar(value: np.ndarray) -> Any:
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return item


def _policy_file(contract_dir: Path, source: str) -> Path:
    path = contract_dir / "baselines" / source / "policy.npz"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Could not find {source.upper()} policy for contract directory {contract_dir}"
    )


def _load_policy_array(path: Path) -> np.ndarray:
    cached = _POLICY_CACHE.get(path)
    if cached is not None:
        _POLICY_CACHE.move_to_end(path)
        return cached
    with np.load(path, allow_pickle=False) as data:
        policies = np.asarray(data["policies"])
    _POLICY_CACHE[path] = policies
    while len(_POLICY_CACHE) > POLICY_CACHE_SIZE:
        _POLICY_CACHE.popitem(last=False)
    return policies


@dataclass
class SupplierBaseline:
    """Stateful supplier baseline replaying solver order policies."""

    baseline: str
    contract_id: str
    contract_dir: Path
    config: dict[str, Any]
    terms: dict[str, Any]
    private_values: dict[str, int] | None = None
    soup_equivalents: dict[str, int] | None = None
    default_min_qty: dict[str, int] | None = None
    environment_id: int = 1
    payment_schedule: dict[int, float] = field(init=False)
    production_schedule: dict[int, dict[str, int]] = field(init=False)
    production_weeks: list[int] = field(init=False)
    production_week_to_idx: dict[int, int] = field(init=False)
    contingencies: set[str] = field(init=False)
    mode: str = field(init=False)
    min_qty_schedule: list[dict[str, int]] = field(init=False)
    deduction_prices_schedule: list[dict[str, int]] = field(init=False)
    rollover_max_deficit: int = field(init=False)
    policy_source: str | None = field(init=False)
    policy_path: Path | None = field(init=False)
    policy_axes: list[str] = field(init=False)
    price_scenarios: list[tuple[int, int, int]] = field(init=False)
    customer_defaulted: bool = False
    customer_may_terminate: bool = False
    contract_terminated: bool = False
    customer_payments_suspended: bool = False
    carry_deficit: dict[str, int] = field(default_factory=_dish_zero)
    total_deficit: dict[str, int] = field(default_factory=_dish_zero)
    had_prior_shortfall: bool = False
    pending_deduction: float = 0.0
    latest_payment: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.baseline not in {"rc", "rcc", "re"}:
            raise ValueError(f"Unknown supplier baseline: {self.baseline}")
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
        self.payment_schedule = _payment_schedule_from_terms(self.terms)
        self.production_schedule = _production_schedule_from_terms(self.terms)
        self.production_weeks = sorted(self.production_schedule)
        self.production_week_to_idx = {
            week: idx for idx, week in enumerate(self.production_weeks)
        }
        self.contingencies = set(self.config.get("contingencies", []) or [])
        self.mode = str(self.config.get("contingency_mode") or CONTINGENCY_GRIM)
        self.min_qty_schedule = _min_qty_schedule(
            self.config,
            self.production_weeks,
            self.default_min_qty,
        )
        self.deduction_prices_schedule = _deduction_price_schedule(
            self.config,
            self.production_weeks,
            self.private_values,
        )
        self.rollover_max_deficit = _rollover_max_deficit(self.config)
        self.policy_source = None
        self.policy_path = None
        self.policy_axes = []
        self.price_scenarios = []
        if self.baseline != "re":
            self.policy_source = "rcc"
            self.policy_path = _policy_file(self.contract_dir, self.policy_source)
            with np.load(self.policy_path, allow_pickle=False) as data:
                self.policy_axes = json.loads(_coerce_scalar(data["policy_axes"]))
                self.price_scenarios = [
                    tuple(int(x) for x in row)
                    for row in np.asarray(data["PRICE_RANGES"]).tolist()
                ]

    @property
    def baseline_type(self) -> str:
        return {
            "rc": "rational_complier_supplier",
            "rcc": "rational_conditional_complier_supplier",
            "re": "rational_exploiter_supplier",
        }[self.baseline]

    @property
    def allow_substitution(self) -> bool:
        return CONTINGENCY_SUBSTITUTION in self.contingencies

    @property
    def allow_rollover(self) -> bool:
        return self.mode in {CONTINGENCY_ROLLOVER, CONTINGENCY_PAYDED_ROLLOVER}

    @property
    def allow_payment_deduction(self) -> bool:
        return self.mode in {CONTINGENCY_PAYDED, CONTINGENCY_PAYDED_ROLLOVER}

    def scheduled_payment(self, week: int) -> float:
        return float(self.payment_schedule.get(int(week), 0.0))

    def observe_payment(
        self,
        *,
        week: int,
        payment: float,
    ) -> dict[str, Any]:
        self.latest_payment = float(payment)
        scheduled = self.scheduled_payment(week)
        max_deduction_available = min(self.pending_deduction, scheduled)
        ordinary_due = scheduled - max_deduction_available
        deduction_taken = min(
            max_deduction_available,
            max(0.0, scheduled - float(payment)),
        )
        if self.contract_terminated:
            amount_due = 0.0
            shortfall = 0.0
            reason = "contract_terminated"
        elif (
            self.customer_may_terminate
            and float(payment) + 1e-6 < ordinary_due
        ):
            self.contract_terminated = True
            self.customer_payments_suspended = True
            amount_due = 0.0
            shortfall = 0.0
            reason = "customer_exercised_termination_after_supplier_default"
        else:
            amount_due = ordinary_due
            shortfall = max(0.0, amount_due - float(payment))
            reason = (
                "payment_deduction_applied"
                if max_deduction_available > 0
                else "scheduled_payment_due"
            )

        if max_deduction_available > 0:
            # The deduction right applies to the next unpaid scheduled payment.
            # Any amount not used on that payment expires rather than carrying.
            self.pending_deduction = 0.0
        if shortfall > 1e-6:
            self.customer_defaulted = True
        record = {
            "event": "payment_observed",
            "week": int(week),
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "scheduled_payment": round(scheduled, 2),
            "gross_due_before_deductions": round(scheduled, 2),
            "max_deduction_available": round(max_deduction_available, 2),
            "deduction_taken": round(deduction_taken, 2),
            "legally_due_payment": round(amount_due, 2),
            "payment": round(float(payment), 2),
            "customer_payment_shortfall": round(shortfall, 2),
            "reason": reason,
            "customer_defaulted": self.customer_defaulted,
            "customer_may_terminate": self.customer_may_terminate,
            "contract_terminated": self.contract_terminated,
            "customer_payments_suspended": self.customer_payments_suspended,
            "pending_deduction": round(self.pending_deduction, 2),
        }
        self.history.append(record)
        return record

    def _price_scenario_idx(self, prices: dict[str, float]) -> int:
        observed = tuple(int(prices[ing]) for ing in INGS)
        try:
            return self.price_scenarios.index(observed)
        except ValueError as exc:
            raise ValueError(f"Unexpected ingredient price scenario: {observed}") from exc

    def _policy_index(self, game: Any, week: int) -> tuple[Any, ...]:
        round_idx = self.production_week_to_idx[int(week)]
        policy_balance = float(game.supplier_balance)
        if not self.allow_payment_deduction:
            policy_balance -= self.latest_payment
        axis_values: dict[str, int] = {
            "round": round_idx,
            "inv_x": int(game.inventory["X"]),
            "inv_y": int(game.inventory["Y"]),
            "inv_z": int(game.inventory["Z"]),
            # Ordinary policies are indexed before the current payment;
            # deduction-aware policies are indexed after it.
            "budget": int(max(0, math.floor(policy_balance))),
            "scenario": self._price_scenario_idx(game.ingredient_prices),
            "had_prior_shortfall": int(self.had_prior_shortfall),
            "deficit_A": int(self.carry_deficit["A"]),
            "deficit_B": int(self.carry_deficit["B"]),
            "deficit_C": int(self.carry_deficit["C"]),
        }
        policies = _load_policy_array(self.policy_path)
        index: list[Any] = []
        for axis, dim in zip(self.policy_axes, policies.shape):
            if axis == "ingredient":
                index.append(slice(None))
            else:
                value = axis_values[axis]
                index.append(max(0, min(int(value), int(dim) - 1)))
        return tuple(index)

    def decide_purchases(self, game: Any, week: int) -> tuple[dict[str, int], dict[str, Any]]:
        if self.baseline == "re":
            decision = _ing_zero()
            record = self._purchase_record(week, decision, "re_never_buys")
            self.history.append(record)
            return decision, record
        if self.contract_terminated:
            decision = _ing_zero()
            record = self._purchase_record(week, decision, "contract_terminated")
            self.history.append(record)
            return decision, record
        if self.baseline == "rcc" and self.customer_defaulted:
            decision = _ing_zero()
            record = self._purchase_record(week, decision, "rcc_grim_after_customer_default")
            self.history.append(record)
            return decision, record

        policies = _load_policy_array(self.policy_path)
        raw = np.asarray(policies[self._policy_index(game, week)], dtype=np.int64)
        decision = {
            ing: max(0, min(int(raw[idx]), game.environment.max_order))
            for idx, ing in enumerate(INGS)
        }
        adjusted = False
        while self._purchase_cost(decision, game.ingredient_prices) > float(game.supplier_balance):
            nonzero = [ing for ing in INGS if decision[ing] > 0]
            if not nonzero:
                break
            ing = max(nonzero, key=lambda key: float(game.ingredient_prices[key]))
            decision[ing] -= 1
            adjusted = True
        reason = f"{self.policy_source}_policy_table"
        if adjusted:
            reason += "_affordability_adjusted"
        record = self._purchase_record(week, decision, reason)
        record["policy_path"] = str(self.policy_path)
        record["policy_source"] = self.policy_source
        record["policy_axes"] = self.policy_axes
        self.history.append(record)
        return decision, record

    def _purchase_cost(self, decision: dict[str, int], prices: dict[str, float]) -> float:
        return float(sum(int(decision[ing]) * float(prices[ing]) for ing in INGS))

    def _purchase_record(
        self,
        week: int,
        decision: dict[str, int],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "event": "purchase_decision",
            "week": int(week),
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "reason": reason,
            **{f"buy_{ing}": int(decision[ing]) for ing in INGS},
            "customer_defaulted": self.customer_defaulted,
            "customer_may_terminate": self.customer_may_terminate,
            "contract_terminated": self.contract_terminated,
            "customer_payments_suspended": self.customer_payments_suspended,
            "carry_deficit": dict(self.carry_deficit),
            "total_deficit": dict(self.total_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
        }

    def decide_production(
        self,
        game: Any,
        week: int,
        *,
        received: dict[str, int] | None = None,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        del received
        if self.baseline == "re":
            decision = _dish_zero()
            round_idx = self.production_week_to_idx[int(week)]
            target = self._effective_target(round_idx)
            exact_req = self._exact_requirement(round_idx, target)
            weights = self._production_weights(round_idx)
            profile = _delivery_profile(
                decision,
                target,
                exact_req,
                allow_substitution=self.allow_substitution,
                require_complete_substitution=(
                    self.allow_payment_deduction or self.allow_rollover
                ),
                weights=weights,
                soup_equivalents=self.soup_equivalents,
                private_values=self.private_values,
            )
            self._advance_contingency_state(profile)
            record = self._production_record(week, decision, profile, "re_never_produces")
            self.history.append(record)
            return decision, record
        if self.contract_terminated:
            decision = _dish_zero()
            record = self._production_record(week, decision, None, "contract_terminated")
            self.history.append(record)
            return decision, record
        if self.baseline == "rcc" and self.customer_defaulted:
            decision = _dish_zero()
            record = self._production_record(
                week,
                decision,
                None,
                "rcc_grim_after_customer_default",
            )
            self.history.append(record)
            return decision, record

        round_idx = self.production_week_to_idx[int(week)]
        target = self._effective_target(round_idx)
        exact_req = self._exact_requirement(round_idx, target)
        weights = self._production_weights(round_idx)
        inventory = {ing: int(game.inventory[ing]) for ing in INGS}

        if self.mode == CONTINGENCY_GRIM and not self.allow_substitution:
            decision = dict(target) if _can_make(target, inventory) else _dish_zero()
            profile = _delivery_profile(
                decision,
                target,
                exact_req,
                allow_substitution=False,
                weights=weights,
                soup_equivalents=self.soup_equivalents,
                private_values=self.private_values,
            )
            self._advance_contingency_state(profile)
            record = self._production_record(
                week, decision, profile, "grim_trigger_exact_fulfillment"
            )
            self.history.append(record)
            return decision, record

        candidates = _production_candidates(
            target,
            exact_req,
            allow_substitution=self.allow_substitution,
            require_complete_substitution=(
                self.allow_payment_deduction or self.allow_rollover
            ),
            weights=weights,
            inventory=inventory,
            soup_equivalents=self.soup_equivalents,
            private_values=self.private_values,
        )
        if (
            (self.mode == CONTINGENCY_GRIM and self.allow_substitution)
            or (self.mode == CONTINGENCY_PAYDED and self.had_prior_shortfall)
        ):
            candidates = [
                candidate for candidate in candidates
                if candidate[1]["full_satisfied"]
            ]
        if self.allow_rollover:
            final_round = round_idx == len(self.production_weeks) - 1
            had_deficit = any(self.carry_deficit.values())
            candidates = [
                candidate for candidate in candidates
                if candidate[1]["full_satisfied"]
                or (
                    not final_round
                    and not had_deficit
                    and all(
                        candidate[1]["shortfall"][dish]
                        <= self.rollover_max_deficit
                        for dish in DISHES
                    )
                )
            ]
        if not candidates:
            decision = _dish_zero()
            profile = _delivery_profile(
                decision,
                target,
                exact_req,
                allow_substitution=self.allow_substitution,
                require_complete_substitution=(
                    self.allow_payment_deduction or self.allow_rollover
                ),
                weights=weights,
                soup_equivalents=self.soup_equivalents,
                private_values=self.private_values,
            )
        else:
            decision, profile = max(
                candidates,
                key=lambda item: self._production_score(item[1]),
            )
        reason = f"{self.mode}_solver_delivery_heuristic"
        self._advance_contingency_state(profile)
        record = self._production_record(week, decision, profile, reason)
        self.history.append(record)
        return decision, record

    def _effective_target(self, round_idx: int) -> dict[str, int]:
        week = self.production_weeks[round_idx]
        target = dict(self.production_schedule[week])
        if self.allow_rollover:
            target = _add(target, self.carry_deficit)
        return target

    def _exact_requirement(self, round_idx: int, target: dict[str, int]) -> dict[str, int]:
        if not self.allow_substitution:
            return dict(target)
        week = self.production_weeks[round_idx]
        base_target = self.production_schedule[week]
        base_min = self.min_qty_schedule[round_idx]
        exact = {
            dish: min(int(base_target[dish]), int(base_min[dish]))
            for dish in DISHES
        }
        if self.allow_rollover:
            exact = _add(exact, self.carry_deficit)
        return exact

    def _production_weights(self, round_idx: int) -> dict[str, int]:
        if self.allow_payment_deduction:
            return dict(self.deduction_prices_schedule[round_idx])
        return dict(self.private_values)

    def _production_score(self, profile: dict[str, Any]) -> tuple[int, int, int]:
        if self.allow_rollover:
            shortfall = profile["shortfall"]
            return (
                1 if profile["full_satisfied"] else 0,
                (
                    -sum(int(shortfall[dish]) for dish in DISHES)
                    if self.mode == CONTINGENCY_ROLLOVER
                    else int(profile["served_weight"])
                ),
                -int(profile["total_usage"]),
            )
        return (
            1 if profile["full_satisfied"] else 0,
            int(profile["served_weight"]),
            -int(profile["total_usage"]),
        )

    def _advance_contingency_state(self, profile: dict[str, Any]) -> None:
        shortfall = dict(profile["shortfall"])
        breach = any(shortfall[dish] > 0 for dish in DISHES)
        if self.allow_payment_deduction:
            self._update_payment_deduction_credit(shortfall, profile)

        if self.mode == CONTINGENCY_GRIM:
            self.customer_may_terminate = self.customer_may_terminate or breach
            return

        if self.mode == CONTINGENCY_PAYDED:
            self.customer_may_terminate = self.customer_may_terminate or (
                self.had_prior_shortfall and breach
            )
            self.had_prior_shortfall = breach
            return

        if self.allow_rollover:
            had_carry = any(self.carry_deficit.values())
            final_round = sum(
                item.get("event") == "production_decision"
                for item in self.history
            ) == len(self.production_weeks) - 1
            rollover_failed = (
                (had_carry and breach)
                or any(
                    shortfall[dish] > self.rollover_max_deficit
                    for dish in DISHES
                )
                or (final_round and breach)
            )
            self.customer_may_terminate = (
                self.customer_may_terminate or rollover_failed
            )
            # The solved policy only covers a carried deficit up to the rollover
            # cap (the "wallet"). Once the caterer defaults on the cap, the true
            # backlog can exceed it, so we keep the full amount in
            # ``total_deficit`` (the "bank") and expose only a capped
            # ``carry_deficit`` (wallet) to the policy index. ``shortfall`` is
            # measured against contract + wallet, so the untouched bank
            # (total - cap) plus this round's shortfall is the new backlog.
            cap = self.rollover_max_deficit
            bank = {
                dish: max(0, int(self.total_deficit[dish]) - cap)
                for dish in DISHES
            }
            self.total_deficit = {
                dish: bank[dish] + int(shortfall[dish])
                for dish in DISHES
            }
            self.carry_deficit = {
                dish: min(int(self.total_deficit[dish]), cap)
                for dish in DISHES
            }
            if self.allow_payment_deduction:
                # BUG-1 fix: the customer deducts the unit-price value of ALL
                # remaining undelivered quantity (the uncapped backlog). The cap
                # on ``carry_deficit`` exists only to index the solved policy
                # table; reusing the capped shortfall for the deduction made the
                # caterer under-predict what the customer would deduct, read the
                # resulting smaller payment as a customer shortfall, and (rcc)
                # spuriously grim-shut-down. Value the uncapped ``total_deficit``
                # so the predicted net payment matches what the customer pays.
                round_idx = max(
                    0,
                    min(
                        len(self.production_weeks) - 1,
                        sum(
                            1
                            for h in self.history
                            if h.get("event") == "production_decision"
                        ),
                    ),
                )
                self.pending_deduction = float(
                    _weighted_value(
                        self.total_deficit,
                        self._production_weights(round_idx),
                    )
                )

    def _update_payment_deduction_credit(
        self,
        shortfall: dict[str, int],
        profile: dict[str, Any],
    ) -> None:
        shortfall_value = float(profile.get("shortfall_value", 0.0))
        if shortfall_value <= 0:
            # Fall back to the active deduction-price weights if the profile was
            # produced before this metadata was attached.
            round_idx = max(
                0,
                min(
                    len(self.production_weeks) - 1,
                    len([h for h in self.history if h.get("event") == "production_decision"]),
                ),
            )
            shortfall_value = float(
                _weighted_value(shortfall, self._production_weights(round_idx))
            )
        self.pending_deduction = shortfall_value

    def _production_record(
        self,
        week: int,
        decision: dict[str, int],
        profile: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        record = {
            "event": "production_decision",
            "week": int(week),
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "reason": reason,
            **{f"produce_{dish}": int(decision[dish]) for dish in DISHES},
            "customer_defaulted": self.customer_defaulted,
            "customer_may_terminate": self.customer_may_terminate,
            "contract_terminated": self.contract_terminated,
            "customer_payments_suspended": self.customer_payments_suspended,
            "carry_deficit": dict(self.carry_deficit),
            "total_deficit": dict(self.total_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
        }
        if profile is not None:
            record.update({
                "served": dict(profile["served"]),
                "shortfall": dict(profile["shortfall"]),
                "full_satisfied": bool(profile["full_satisfied"]),
                "served_weight": int(profile["served_weight"]),
                "shortfall_value": float(profile.get("shortfall_value", 0.0)),
                "produced_soup": int(profile["produced_soup"]),
                "produced_value": int(profile["produced_value"]),
                "total_usage": int(profile["total_usage"]),
            })
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "baseline_type": self.baseline_type,
            "environment_id": self.environment_id,
            "policy_source": self.policy_source,
            "policy_path": str(self.policy_path) if self.policy_path else None,
            "customer_defaulted": self.customer_defaulted,
            "customer_may_terminate": self.customer_may_terminate,
            "contract_terminated": self.contract_terminated,
            "customer_payments_suspended": self.customer_payments_suspended,
            "carry_deficit": dict(self.carry_deficit),
            "total_deficit": dict(self.total_deficit),
            "had_prior_shortfall": self.had_prior_shortfall,
            "pending_deduction": round(self.pending_deduction, 2),
            "history": self.history,
        }
