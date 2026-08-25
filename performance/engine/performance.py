"""Run a contract through the customer and supplier baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_utils import DEFAULT_ENVIRONMENT_ROOT

from .environment import DISHES, INGREDIENTS, EnvironmentEngine


@dataclass(frozen=True)
class PerformanceResult:
    payment_history: list[dict[str, Any]]
    production_history: list[dict[str, Any]]
    total_customer_spent: float
    customer_remaining_budget: float
    supplier_final_balance: float
    delivered: dict[str, int]
    leftover_inventory: dict[str, int]
    rounds_target_met: int
    targets: dict[str, int]
    customer_summary: dict[str, Any]
    supplier_summary: dict[str, Any]
    environment_id: int
    environment_name: str
    environment_axes: dict[str, str]


def _production_schedule(
    terms: dict[str, Any],
) -> dict[int, dict[str, int]]:
    return {
        int(row["week"]): {
            dish: int(row.get(dish, 0)) for dish in DISHES
        }
        for row in terms.get("production_schedule", [])
    }


def run_performance(
    *,
    customer: Any,
    supplier: Any,
    terms: dict[str, Any],
    environment_id: int,
    seed: int = 42,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> PerformanceResult:
    """Execute the canonical payment/production schedule once."""
    game = EnvironmentEngine(environment_id, seed, environment_root)
    payment_history: list[dict[str, Any]] = []
    production_history: list[dict[str, Any]] = []
    total_paid = 0.0

    for week in range(1, game.total_weeks + 1):
        if week in game.payment_weeks:
            payment, customer_decision = customer.decide_payment(
                week=week,
                remaining_budget=game.customer_budget,
            )
            game.apply_payment(payment)
            total_paid += payment
            supplier_observation = supplier.observe_payment(
                week=week,
                payment=payment,
            )
            payment_history.append(
                {
                    "week": week,
                    "amount": round(payment, 2),
                    "customer_decision": customer_decision,
                    "supplier_observation": supplier_observation,
                    "customer_budget": round(game.customer_budget, 2),
                    "supplier_balance": round(game.supplier_balance, 2),
                }
            )
            continue

        if week in game.production_weeks:
            spoilage = game.apply_spoilage()
            purchase, purchase_decision = supplier.decide_purchases(game, week)
            cost, ordered, received = game.apply_purchase(purchase)
            production, production_decision = supplier.decide_production(
                game,
                week,
            )
            produced = game.apply_production(production)
            customer_observation = customer.observe_production(
                week=week,
                produced=produced,
            )
            production_history.append(
                {
                    "week": week,
                    "ingredient_prices": dict(game.ingredient_prices),
                    "ordered": ordered,
                    "received": received,
                    "lost": {
                        ingredient: ordered[ingredient] - received[ingredient]
                        for ingredient in INGREDIENTS
                    },
                    "spoilage": spoilage,
                    "produced": produced,
                    "purchase_cost": round(cost, 2),
                    "supplier_balance": round(game.supplier_balance, 2),
                    "inventory_after": dict(game.inventory),
                    "customer_observation": customer_observation,
                    "supplier_purchase_decision": purchase_decision,
                    "supplier_production_decision": production_decision,
                }
            )
            game.roll_prices()

    schedule = _production_schedule(terms)
    rounds_target_met = sum(
        1
        for row in production_history
        if all(
            row["produced"][dish]
            >= schedule.get(row["week"], {}).get(dish, 0)
            for dish in DISHES
        )
    )
    targets = {
        dish: sum(row[dish] for row in schedule.values()) for dish in DISHES
    }
    environment = game.environment
    return PerformanceResult(
        payment_history=payment_history,
        production_history=production_history,
        total_customer_spent=round(total_paid, 2),
        customer_remaining_budget=round(game.customer_budget, 2),
        supplier_final_balance=round(game.supplier_balance, 2),
        delivered=dict(game.delivered),
        leftover_inventory=dict(game.inventory),
        rounds_target_met=rounds_target_met,
        targets=targets,
        customer_summary=customer.summary(),
        supplier_summary=supplier.summary(),
        environment_id=environment.environment_id,
        environment_name=environment.environment_name,
        environment_axes=dict(environment.environment_axes),
    )
