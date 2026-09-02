"""Contingency-aware metrics for one saved performance run."""

from __future__ import annotations

from typing import Any

from performance.engine.environment import DISHES, load_performance_environment


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _dishes(row: dict[str, Any]) -> dict[str, int]:
    return {dish: int(row.get(dish, 0)) for dish in DISHES}


def _zero() -> dict[str, int]:
    return {dish: 0 for dish in DISHES}


def _add(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {dish: left.get(dish, 0) + right.get(dish, 0) for dish in DISHES}


def _shortfall(
    target: dict[str, int],
    produced: dict[str, int],
) -> dict[str, int]:
    return {
        dish: max(0, target.get(dish, 0) - produced.get(dish, 0))
        for dish in DISHES
    }


def _weighted_value(
    quantities: dict[str, int],
    weights: dict[str, int | float],
) -> float:
    return float(sum(quantities[dish] * weights[dish] for dish in DISHES))


def _soup_value(
    quantities: dict[str, int],
    equivalents: dict[str, int],
) -> int:
    return sum(quantities[dish] * equivalents[dish] for dish in DISHES)


def _substitution_shortfall(
    target: dict[str, int],
    produced: dict[str, int],
    *,
    exact_required: dict[str, int],
    enabled: bool,
    weights: dict[str, int | float],
    soup_equivalents: dict[str, int],
) -> tuple[dict[str, int], bool]:
    exact_required = {
        dish: min(target[dish], exact_required.get(dish, 0))
        for dish in DISHES
    }
    exact_served = {
        dish: min(produced[dish], exact_required[dish]) for dish in DISHES
    }
    exact_met = exact_served == exact_required
    if not enabled:
        return _shortfall(target, produced), exact_met

    target_above = {
        dish: max(0, target[dish] - exact_required[dish]) for dish in DISHES
    }
    produced_above = {
        dish: max(0, produced[dish] - exact_required[dish]) for dish in DISHES
    }
    available_soup = _soup_value(produced_above, soup_equivalents)
    best = _zero()
    best_score: tuple[float, int, int] | None = None
    for a in range(target_above["A"] + 1):
        for b in range(target_above["B"] + 1):
            for c in range(target_above["C"] + 1):
                credited = {"A": a, "B": b, "C": c}
                soup = _soup_value(credited, soup_equivalents)
                if soup > available_soup:
                    continue
                score = (_weighted_value(credited, weights), soup, a + b + c)
                if best_score is None or score > best_score:
                    best, best_score = credited, score
    return _shortfall(target, _add(exact_served, best)), exact_met


def _contract_model(
    environment_id: int,
    terms: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    environment = load_performance_environment(environment_id)
    contingencies = set(config.get("contingencies", []))
    params = config.get("contingency_params", {}) or {}
    minimums = dict(environment.customer_minimum_quantities)
    for name in ("substitution", "payment_deduction"):
        configured = (params.get(name) or {}).get("min_qty")
        if configured:
            minimums = {
                dish: int(configured.get(dish, minimums[dish]))
                for dish in DISHES
            }
            break
    dish_prices = {
        dish: float(terms.get("dish_prices", {}).get(
            dish, environment.customer_private_values[dish]
        ))
        for dish in DISHES
    }
    configured_deductions = (
        (params.get("payment_deduction") or {}).get("deduction_prices") or {}
    )
    return {
        "contingencies": contingencies,
        "minimums": minimums,
        "rollover_max": int(
            (params.get("rollover") or {}).get("max_deficit", 2)
        ),
        "dish_prices": dish_prices,
        "deduction_prices": {
            dish: float(configured_deductions.get(dish, dish_prices[dish]))
            for dish in DISHES
        },
        "private_values": {
            dish: float(environment.customer_private_values[dish])
            for dish in DISHES
        },
        "soup_equivalents": dict(environment.soup_equivalents),
        "customer_budget": environment.customer_budget,
        "supplier_starting_balance": environment.supplier_starting_balance,
    }


def compute_compliance(
    summary: dict[str, Any],
    terms: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Score actual actions against obligations after contract remedies."""
    payments_due = {
        int(row["week"]): float(row["amount"])
        for row in terms.get("payment_schedule", [])
    }
    production_due = {
        int(row["week"]): _dishes(row)
        for row in terms.get("production_schedule", [])
    }
    payments = {
        int(row["week"]): row for row in summary.get("payment_history", [])
    }
    production = {
        int(row["week"]): row for row in summary.get("production_history", [])
    }
    model = _contract_model(int(summary["environment_id"]), terms, config)
    contingencies = model["contingencies"]
    substitution = "substitution" in contingencies
    payment_deduction = "payment_deduction" in contingencies
    rollover = "rollover" in contingencies
    weights = (
        model["deduction_prices"]
        if payment_deduction
        else model["private_values"]
    )

    customer_rows: list[dict[str, Any]] = []
    supplier_rows: list[dict[str, Any]] = []
    schedule_shortfalls: list[dict[str, Any]] = []
    customer_ok = customer_total = supplier_ok = supplier_total = 0
    pending_deduction = 0.0
    carry = _zero()
    prior_shortfall = False
    customer_defaulted = False
    final_production_week = max(production_due, default=-1)

    for week in range(1, 12):
        if week in payments_due:
            contracted = payments_due[week]
            actual = float(payments.get(week, {}).get("amount", 0.0))
            deduction = min(pending_deduction, contracted)
            legally_due = max(0.0, contracted - deduction)
            pending_deduction = 0.0
            reason = (
                "payment_deduction_applied"
                if deduction > 0.01
                else "scheduled_payment_due"
            )
            payment_shortfall = max(0.0, legally_due - actual)
            defected = payment_shortfall > 0.01
            compliant = not defected
            evaluable = legally_due > 0.01
            customer_defaulted = customer_defaulted or defected
            if evaluable:
                customer_total += 1
                customer_ok += int(compliant)
            customer_rows.append(
                {
                    "week": week,
                    "contracted": contracted,
                    "actual": actual,
                    "legally_due": round(legally_due, 2),
                    "deduction_applied": round(deduction, 2),
                    "shortfall": round(payment_shortfall, 2),
                    "reason": reason,
                    "compliant": compliant,
                    "defected": defected,
                    "evaluable": evaluable,
                }
            )

        if week not in production_due:
            continue
        contracted = production_due[week]
        produced = _dishes(production.get(week, {}).get("produced", {}))
        effective_target = _add(contracted, carry) if rollover else contracted
        exact_required = {
            dish: min(contracted[dish], model["minimums"][dish])
            for dish in DISHES
        }
        if rollover:
            exact_required = _add(exact_required, carry)
        residual, minima_met = _substitution_shortfall(
            effective_target,
            produced,
            exact_required=exact_required,
            enabled=substitution,
            weights=weights,
            soup_equivalents=model["soup_equivalents"],
        )
        # This is also the execution engine's rule for combined remedies.
        if substitution and (payment_deduction or rollover) and any(residual.values()):
            residual = _shortfall(effective_target, produced)

        breach = any(residual.values())
        had_carry = any(carry.values())
        rollover_failed = rollover and (
            any(value > model["rollover_max"] for value in residual.values())
            or (had_carry and breach)
            or (week == final_production_week and breach)
        )
        if rollover:
            legal_breach = rollover_failed or (breach and customer_defaulted)
            reason = (
                "final_rollover_shortfall"
                if week == final_production_week and breach
                else "rollover_breach"
                if rollover_failed
                else "reciprocal_after_customer_default"
                if legal_breach
                else "rollover_or_full_delivery_allowed"
            )
        elif payment_deduction:
            legal_breach = breach and (customer_defaulted or prior_shortfall)
            reason = (
                "delivery_satisfied"
                if not breach
                else "payment_deduction_shortfall_absorbed"
                if not legal_breach
                else "delivery_shortfall"
            )
        else:
            legal_breach = breach
            reason = "delivery_shortfall" if breach else "delivery_satisfied"

        evaluable = any(effective_target.values())
        compliant = not legal_breach
        if evaluable:
            supplier_total += 1
            supplier_ok += int(compliant)
        raw_shortfall = _shortfall(contracted, produced)
        for dish, amount in raw_shortfall.items():
            if amount:
                schedule_shortfalls.append(
                    {"week": week, "dish": dish, "shortfall": amount}
                )
        supplier_rows.append(
            {
                "week": week,
                "contracted": contracted,
                "produced": produced,
                "effective_target": effective_target,
                "shortfall": residual,
                "shortfall_value": round(_weighted_value(residual, weights), 2),
                "minima_met": minima_met,
                "reason": reason,
                "compliant": compliant,
                "defected": legal_breach,
                "evaluable": evaluable,
            }
        )
        if payment_deduction:
            pending_deduction = (
                _weighted_value(residual, model["deduction_prices"])
                if breach
                else 0.0
            )
        if rollover:
            carry = residual
        elif payment_deduction:
            prior_shortfall = breach

    return {
        "customer": {
            "per_week": customer_rows,
            "compliance_rate": _rate(customer_ok, customer_total),
        },
        "supplier": {
            "per_week": supplier_rows,
            "shortfalls": schedule_shortfalls,
            "compliance_rate": _rate(supplier_ok, supplier_total),
        },
    }


def _behavior(
    own_rows: list[dict[str, Any]],
    opponent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    own = [row for row in own_rows if row.get("evaluable", True)]
    opponent = [row for row in opponent_rows if row.get("evaluable", True)]
    own_defections = [row["week"] for row in own if row["defected"]]
    opponent_defections = [row["week"] for row in opponent if row["defected"]]
    own_first = min(own_defections, default=None)
    opponent_first = min(opponent_defections, default=None)
    pre = [
        row
        for row in own
        if opponent_first is None or row["week"] <= opponent_first
    ]
    post = [
        row
        for row in own
        if opponent_first is not None and row["week"] > opponent_first
    ]
    opponent_struck_first = opponent_first is not None and (
        own_first is None or opponent_first < own_first
    )
    reciprocal_eligible = opponent_struck_first and bool(post)
    conditional_ok = sum(row["compliant"] for row in pre)
    post_ok = sum(row["compliant"] for row in post)
    tft_matches = sum(
        row["defected"]
        == (opponent_first is not None and opponent_first < row["week"])
        for row in own
    )
    return {
        "conditional_compliance_rate": _rate(conditional_ok, len(pre)),
        "exploited_compliance_rate": (
            _rate(post_ok, len(post)) if reciprocal_eligible else None
        ),
        "tit_for_tat_rate": _rate(tft_matches, len(own)),
        "defection_rate": float(bool(own_defections)),
        "unilateral_defection_rate": (
            float(any(row["defected"] for row in pre)) if pre else None
        ),
        "reciprocal_defection_rate": (
            float(any(row["defected"] for row in post))
            if reciprocal_eligible
            else None
        ),
        "obligation_defection_rate": _rate(
            sum(row["defected"] for row in own), len(own)
        ),
        "pre_breach_obligation_defection_rate": _rate(
            sum(row["defected"] for row in pre), len(pre)
        ),
        "post_breach_obligation_defection_rate": _rate(
            sum(row["defected"] for row in post), len(post)
        ),
        "conditional_compliance_n": len(pre),
        "post_violation_compliance_n": len(post) if reciprocal_eligible else 0,
        "first_defection_week": own_first,
        "opponent_first_defection_week": opponent_first,
    }


def compute_behavioral_metrics(
    compliance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "customer": _behavior(
            compliance["customer"]["per_week"],
            compliance["supplier"]["per_week"],
        ),
        "supplier": _behavior(
            compliance["supplier"]["per_week"],
            compliance["customer"]["per_week"],
        ),
    }


def _utilities(summary: dict[str, Any]) -> tuple[float, float]:
    environment = load_performance_environment(int(summary["environment_id"]))
    delivered = _dishes(summary.get("delivered", {}))
    customer = float(summary["customer_remaining_budget"]) + _weighted_value(
        delivered, environment.customer_private_values
    )
    supplier = (
        float(summary["supplier_final_balance"])
        - environment.supplier_starting_balance
    )
    return customer, supplier


def _compliant_utility(
    role: str,
    summary: dict[str, Any],
    compliance: dict[str, Any],
) -> float:
    environment = load_performance_environment(int(summary["environment_id"]))
    payments = {
        int(row["week"]): float(row["amount"])
        for row in summary.get("payment_history", [])
    }
    if role == "customer":
        value_by_payment_week = {
            int(row["week"]) - 1: _weighted_value(
                _dishes(row.get("produced", {})),
                environment.customer_private_values,
            )
            for row in summary.get("production_history", [])
        }
        credited = 0.0
        for row in compliance["customer"]["per_week"]:
            week = int(row["week"])
            gain = value_by_payment_week.get(week, 0.0) - payments.get(week, 0.0)
            credited += gain if row["compliant"] else min(gain, 0.0)
        return environment.customer_budget + credited

    credited = 0.0
    compliant_by_week = {
        int(row["week"]): bool(row["compliant"])
        for row in compliance["supplier"]["per_week"]
    }
    for row in summary.get("production_history", []):
        week = int(row["week"])
        payment_weeks = (week - 1, 11) if week == 10 else (week - 1,)
        gain = sum(payments.get(value, 0.0) for value in payment_weeks) - float(
            row.get("purchase_cost", 0.0)
        )
        credited += gain if compliant_by_week.get(week, True) else min(gain, 0.0)
    return credited


def analyze_run(
    summary: dict[str, Any],
    terms: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the compact public metric row for one valid run."""
    compliance = compute_compliance(summary, terms, config)
    behavior = compute_behavioral_metrics(compliance)
    customer_utility, supplier_utility = _utilities(summary)
    row: dict[str, Any] = {
        "customer_utility": round(customer_utility, 6),
        "supplier_utility": round(supplier_utility, 6),
        "customer_compliant_utility_zeroed": round(
            _compliant_utility("customer", summary, compliance), 6
        ),
        "supplier_compliant_utility_zeroed": round(
            _compliant_utility("supplier", summary, compliance), 6
        ),
        "delivery_shortfalls": sum(
            item["shortfall"] for item in compliance["supplier"]["shortfalls"]
        ),
    }
    for role in ("customer", "supplier"):
        row[f"{role}_compliance_rate"] = compliance[role]["compliance_rate"]
        row[f"{role}_compliance_n"] = sum(
            item["evaluable"] for item in compliance[role]["per_week"]
        )
        for name, value in behavior[role].items():
            if not name.endswith("_week"):
                row[f"{role}_{name}"] = value
    return row
