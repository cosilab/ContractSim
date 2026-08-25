"""Environment-backed state engine for contract performance."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_utils import DEFAULT_ENVIRONMENT_ROOT, load_environment


DISHES = ("A", "B", "C")
INGREDIENTS = ("X", "Y", "Z")


@dataclass(frozen=True)
class PerformanceEnvironment:
    """Execution parameters resolved from one canonical environment."""

    environment_id: int
    environment_name: str
    environment_axes: dict[str, str]
    customer_budget: float
    supplier_starting_balance: float
    total_weeks: int
    payment_weeks: tuple[int, ...]
    production_weeks: tuple[int, ...]
    recipes: dict[str, dict[str, int]]
    initial_inventory: dict[str, int]
    max_inventory: int
    max_order: int
    price_distributions: dict[str, tuple[tuple[float, float], ...]]
    delivery_minimum_received: str
    spoilage_distribution: tuple[tuple[int, float], ...]
    resample_prices: bool
    customer_private_values: dict[str, int]
    soup_equivalents: dict[str, int]
    customer_minimum_quantities: dict[str, int]

    def minimum_received(self, ordered: int) -> int:
        expression = self.delivery_minimum_received
        if expression == "ordered":
            return ordered
        if expression == "floor(3*ordered/4)":
            return math.floor(3 * ordered / 4)
        if expression == "floor(ordered/2)":
            return ordered // 2
        raise ValueError(
            f"Unsupported delivery minimum expression: {expression!r}"
        )


def _weighted_rows(
    rows: list[dict[str, Any]],
    value_key: str,
) -> tuple[tuple[float, float], ...]:
    values = tuple(
        (float(row[value_key]), float(row["probability"])) for row in rows
    )
    total = sum(probability for _value, probability in values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Distribution probabilities sum to {total}, expected 1")
    return values


def load_performance_environment(
    environment_id: int,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> PerformanceEnvironment:
    """Load the execution fields used by the performance engine."""
    payload = load_environment(environment_id, environment_root)
    base = payload["base_game"]
    operating = payload["operating_conditions"]
    inventory = base["inventory"]
    schedule = base["schedule"]
    preference = payload["customer_preference"]

    prices = {
        ingredient: _weighted_rows(spec["outcomes"], "value")
        for ingredient, spec in operating["prices"]["ingredients"].items()
    }
    spoilage = tuple(
        (int(value), probability)
        for value, probability in _weighted_rows(
            operating["spoilage"]["outcomes"],
            "units_lost",
        )
    )
    recipes = {
        dish: {
            ingredient: int(quantity)
            for ingredient, quantity in recipe.items()
            if int(quantity) != 0
        }
        for dish, recipe in base["recipes"].items()
    }

    return PerformanceEnvironment(
        environment_id=environment_id,
        environment_name=str(payload["environment_name"]),
        environment_axes={
            str(key): str(value) for key, value in payload["axes"].items()
        },
        customer_budget=float(base["customer"]["starting_budget"]),
        supplier_starting_balance=float(
            payload["caterer_capital"]["starting_balance"]
        ),
        total_weeks=int(schedule["total_weeks"]),
        payment_weeks=tuple(int(week) for week in schedule["payment_weeks"]),
        production_weeks=tuple(
            int(week) for week in schedule["production_weeks"]
        ),
        recipes=recipes,
        initial_inventory={
            ingredient: int(quantity)
            for ingredient, quantity in inventory["initial_units"].items()
        },
        max_inventory=int(inventory["max_units_per_ingredient"]),
        max_order=int(inventory["max_order_per_ingredient_per_round"]),
        price_distributions=prices,
        delivery_minimum_received=str(
            operating["delivery"]["minimum_received"]
        ),
        spoilage_distribution=spoilage,
        resample_prices=bool(
            operating["prices"]["resampled_each_production_round"]
        ),
        customer_private_values={
            dish: int(value)
            for dish, value in preference["private_values"].items()
        },
        soup_equivalents={
            dish: int(value)
            for dish, value in preference[
                "substitution_soup_equivalents"
            ].items()
        },
        customer_minimum_quantities={
            dish: int(value)
            for dish, value in base["customer"][
                "minimum_dishes_per_production_round"
            ].items()
        },
    )


class EnvironmentEngine:
    """Mutable state and stochastic transitions for one performance run."""

    def __init__(
        self,
        environment_id: int,
        seed: int = 42,
        environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
    ) -> None:
        self.environment = load_performance_environment(
            environment_id,
            environment_root,
        )
        self.total_weeks = self.environment.total_weeks
        self.payment_weeks = self.environment.payment_weeks
        self.production_weeks = self.environment.production_weeks
        self.customer_starting_budget = self.environment.customer_budget
        self.supplier_starting_balance = (
            self.environment.supplier_starting_balance
        )
        self.customer_budget = self.customer_starting_budget
        self.supplier_balance = self.supplier_starting_balance
        self.inventory = dict(self.environment.initial_inventory)
        self.delivered = {dish: 0 for dish in DISHES}
        self._rng = random.Random(seed)
        self.ingredient_prices: dict[str, float] = {}
        self.roll_prices(force=True)

    def _draw_weighted(self, rows: tuple[tuple[float, float], ...]) -> float:
        values = [value for value, _probability in rows]
        weights = [probability for _value, probability in rows]
        if len(set(weights)) == 1:
            return self._rng.choice(values)
        return self._rng.choices(values, weights=weights, k=1)[0]

    def roll_prices(self, *, force: bool = False) -> None:
        if (
            self.ingredient_prices
            and not self.environment.resample_prices
            and not force
        ):
            return
        self.ingredient_prices = {
            ingredient: self._draw_weighted(distribution)
            for ingredient, distribution in (
                self.environment.price_distributions.items()
            )
        }

    def apply_payment(self, amount: float) -> None:
        if amount < 0 or amount > self.customer_budget:
            raise ValueError("payment must be within the customer's budget")
        self.customer_budget -= amount
        self.supplier_balance += amount

    def apply_spoilage(self) -> dict[str, int]:
        lost: dict[str, int] = {}
        for ingredient in INGREDIENTS:
            quantity = self.inventory[ingredient]
            loss = (
                min(
                    quantity,
                    int(self._draw_weighted(self.environment.spoilage_distribution)),
                )
                if quantity > 0
                else 0
            )
            self.inventory[ingredient] -= loss
            lost[ingredient] = loss
        return lost

    def purchase_cost(self, purchase: dict[str, int]) -> float:
        return float(
            sum(
                int(purchase[ingredient]) * self.ingredient_prices[ingredient]
                for ingredient in INGREDIENTS
            )
        )

    def apply_purchase(
        self,
        purchase: dict[str, int],
    ) -> tuple[float, dict[str, int], dict[str, int]]:
        ordered = {
            ingredient: int(purchase.get(ingredient, 0))
            for ingredient in INGREDIENTS
        }
        if any(
            quantity < 0 or quantity > self.environment.max_order
            for quantity in ordered.values()
        ):
            raise ValueError("purchase quantities must be within the order limit")
        cost = self.purchase_cost(ordered)
        if cost > self.supplier_balance:
            raise ValueError("purchase cost exceeds the supplier balance")

        self.supplier_balance -= cost
        received = {
            ingredient: (
                self._rng.randint(
                    self.environment.minimum_received(ordered[ingredient]),
                    ordered[ingredient],
                )
                if ordered[ingredient] > 0
                else 0
            )
            for ingredient in INGREDIENTS
        }
        for ingredient, quantity in received.items():
            self.inventory[ingredient] = min(
                self.inventory[ingredient] + quantity,
                self.environment.max_inventory,
            )
        return cost, ordered, received

    def apply_production(self, production: dict[str, int]) -> dict[str, int]:
        produced = {
            dish: int(production.get(dish, 0)) for dish in DISHES
        }
        if any(quantity < 0 for quantity in produced.values()):
            raise ValueError("production quantities must be nonnegative")

        required = {ingredient: 0 for ingredient in INGREDIENTS}
        for dish, quantity in produced.items():
            for ingredient, units in self.environment.recipes[dish].items():
                required[ingredient] += quantity * units
        if any(
            required[ingredient] > self.inventory[ingredient]
            for ingredient in INGREDIENTS
        ):
            raise ValueError("production exceeds available inventory")

        for ingredient, quantity in required.items():
            self.inventory[ingredient] -= quantity
        for dish, quantity in produced.items():
            self.delivered[dish] += quantity
        return produced
