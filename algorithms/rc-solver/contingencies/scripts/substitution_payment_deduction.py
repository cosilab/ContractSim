"""Joint substitution and payment-deduction implementation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from types import MappingProxyType
from typing import Mapping

import numpy as np

from environment import Environment
from transitions import PriceKernel

from .payment_deduction import (
    PaymentDeductionContingency,
    PaymentDeductionOneStepSolver,
    _integer_mapping,
)
from .substitution import SubstitutionContingency


@dataclass(frozen=True)
class SubstitutionPaymentDeductionOutcome:
    """Selected production after substitution and deduction scoring."""

    production: Mapping[str, int]
    ingredient_usage: Mapping[str, int]
    leftover_inventory: Mapping[str, int]
    served: Mapping[str, int]
    shortfall: Mapping[str, int]
    weighted_shortfall: float
    complete_fulfilment: bool
    exact_minimums_satisfied: bool
    substitution_points_created: int
    substitution_points_used: int
    customer_value: float

    @property
    def satisfied(self) -> bool:
        return self.complete_fulfilment

    @property
    def remaining_shortfall(self) -> Mapping[str, int]:
        return self.shortfall


@dataclass(frozen=True, eq=False)
class SubstitutionPaymentDeductionRoundOutcomes:
    """Dense outcome lookup with a leading previous-shortfall axis."""

    production: np.ndarray
    ingredient_usage: np.ndarray
    leftover_inventory: np.ndarray
    served: np.ndarray
    shortfall: np.ndarray
    weighted_shortfall: np.ndarray
    complete_fulfilment: np.ndarray
    customer_value: np.ndarray


class SubstitutionPaymentDeductionContingency:
    """Apply substitution first and price only its residual shortfall."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        self.environment = environment
        self.product_names = environment.supplier.products
        self.resource_names = environment.supplier.resources
        self.substitution = SubstitutionContingency(
            environment,
            minimum_quantities=minimum_quantities,
        )
        payment_deduction = PaymentDeductionContingency(
            environment,
            deduction_prices,
        )
        self.minimum_quantities = self.substitution.minimum_quantities
        self.deduction_prices = payment_deduction.deduction_prices

    def _ingredient_usage(self, production: Mapping[str, int]) -> dict[str, int]:
        recipes = self.environment.supplier.recipes
        return {
            resource: sum(
                production[dish] * recipes[dish].get(resource, 0)
                for dish in self.product_names
            )
            for resource in self.resource_names
        }

    def _allocate_points(
        self,
        capacities: Mapping[str, int],
        available_points: int,
    ) -> tuple[dict[str, int], int]:
        """Allocate points to avoid the greatest deduction, then use more credit."""
        point_values = self.environment.customer.substitution_rules
        best_quantities = (0,) * len(self.product_names)
        best_key: tuple[float, int, int] = (0.0, 0, 0)
        ranges = (range(capacities[dish] + 1) for dish in self.product_names)
        for quantities in product(*ranges):
            points = sum(
                quantity * point_values[dish]
                for dish, quantity in zip(self.product_names, quantities)
            )
            if points > available_points:
                continue
            deduction_avoided = sum(
                quantity * self.deduction_prices[dish]
                for dish, quantity in zip(self.product_names, quantities)
            )
            key = (deduction_avoided, points, sum(quantities))
            if key > best_key:
                best_key = key
                best_quantities = quantities
        return dict(zip(self.product_names, best_quantities)), best_key[1]

    def evaluate_production(
        self,
        target_quantities: Mapping[str, int],
        production: Mapping[str, int],
        available_inventory: Mapping[str, int] | None = None,
    ) -> SubstitutionPaymentDeductionOutcome:
        """Evaluate one physical plan under both clauses."""
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        produced = _integer_mapping(
            production,
            self.product_names,
            label="production",
        )
        usage = self._ingredient_usage(produced)

        if available_inventory is None:
            leftovers: dict[str, int] = {}
        else:
            inventory = _integer_mapping(
                available_inventory,
                self.resource_names,
                label="available_inventory",
            )
            if any(usage[name] > inventory[name] for name in self.resource_names):
                raise ValueError("production exceeds available inventory")
            leftovers = {
                name: inventory[name] - usage[name] for name in self.resource_names
            }

        exact_minimums = {
            dish: min(targets[dish], self.minimum_quantities[dish])
            for dish in self.product_names
        }
        exact_served = {
            dish: min(produced[dish], exact_minimums[dish])
            for dish in self.product_names
        }
        point_values = self.environment.customer.substitution_rules
        points_created = sum(
            max(produced[dish] - exact_minimums[dish], 0) * point_values[dish]
            for dish in self.product_names
        )
        above_minimum_demand = {
            dish: targets[dish] - exact_minimums[dish]
            for dish in self.product_names
        }
        credited_above_minimum, points_used = self._allocate_points(
            above_minimum_demand,
            points_created,
        )
        substituted_served = {
            dish: exact_served[dish] + credited_above_minimum[dish]
            for dish in self.product_names
        }
        substituted_shortfall = {
            dish: targets[dish] - substituted_served[dish]
            for dish in self.product_names
        }
        if all(value == 0 for value in substituted_shortfall.values()):
            served = substituted_served
            shortfall = substituted_shortfall
            complete = True
        else:
            served = {
                dish: min(produced[dish], targets[dish])
                for dish in self.product_names
            }
            shortfall = {
                dish: targets[dish] - served[dish]
                for dish in self.product_names
            }
            points_used = 0
            complete = False
        weighted_shortfall = sum(
            shortfall[dish] * self.deduction_prices[dish]
            for dish in self.product_names
        )
        customer_value = sum(
            produced[dish] * self.environment.customer.private_values[dish]
            for dish in self.product_names
        )
        return SubstitutionPaymentDeductionOutcome(
            production=MappingProxyType(produced),
            ingredient_usage=MappingProxyType(usage),
            leftover_inventory=MappingProxyType(leftovers),
            served=MappingProxyType(served),
            shortfall=MappingProxyType(shortfall),
            weighted_shortfall=float(weighted_shortfall),
            complete_fulfilment=complete,
            exact_minimums_satisfied=all(
                produced[dish] >= exact_minimums[dish]
                for dish in self.product_names
            ),
            substitution_points_created=points_created,
            substitution_points_used=points_used,
            customer_value=float(customer_value),
        )

    def select_production(
        self,
        target_quantities: Mapping[str, int],
        available_inventory: Mapping[str, int],
        *,
        previous_shortfall: bool = False,
    ) -> SubstitutionPaymentDeductionOutcome:
        """Select full service, then least deduction, then least usage."""
        if not isinstance(previous_shortfall, bool):
            raise TypeError("previous_shortfall must be a bool")
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        inventory = _integer_mapping(
            available_inventory,
            self.resource_names,
            label="available_inventory",
        )
        exact_minimums = {
            dish: min(targets[dish], self.minimum_quantities[dish])
            for dish in self.product_names
        }
        point_values = self.environment.customer.substitution_rules
        above_minimum_points = sum(
            (targets[dish] - exact_minimums[dish]) * point_values[dish]
            for dish in self.product_names
        )
        maximum_production = {
            dish: exact_minimums[dish]
            + (
                ceil(above_minimum_points / point_values[dish])
                if above_minimum_points
                else 0
            )
            for dish in self.product_names
        }

        best_outcome: SubstitutionPaymentDeductionOutcome | None = None
        best_rank: tuple[bool, float, int] | None = None
        ranges = (
            range(maximum_production[dish] + 1) for dish in self.product_names
        )
        for quantities in product(*ranges):
            plan = dict(zip(self.product_names, quantities))
            usage = self._ingredient_usage(plan)
            if any(usage[name] > inventory[name] for name in self.resource_names):
                continue
            outcome = self.evaluate_production(targets, plan, inventory)
            rank = (
                outcome.complete_fulfilment,
                -outcome.weighted_shortfall,
                -sum(outcome.ingredient_usage.values()),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_outcome = outcome

        assert best_outcome is not None
        if previous_shortfall and not best_outcome.complete_fulfilment:
            return self.evaluate_production(
                targets,
                {dish: 0 for dish in self.product_names},
                inventory,
            )
        return best_outcome

    def precompute_round(
        self,
        target_quantities: Mapping[str, int],
    ) -> SubstitutionPaymentDeductionRoundOutcomes:
        """Compile both hardship cases for every delivered inventory state."""
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        number_of_inventory_states = (
            self.environment.supplier.max_inventory_per_resource + 1
        )
        inventory_shape = (number_of_inventory_states,) * len(self.resource_names)
        hardship_shape = (2,) + inventory_shape
        production = np.zeros(
            hardship_shape + (len(self.product_names),), dtype=np.int16
        )
        ingredient_usage = np.zeros(
            hardship_shape + (len(self.resource_names),), dtype=np.int16
        )
        leftover_inventory = np.zeros_like(ingredient_usage)
        served = np.zeros_like(production)
        shortfall = np.zeros_like(production)
        weighted_shortfall = np.zeros(hardship_shape, dtype=np.float32)
        complete_fulfilment = np.zeros(hardship_shape, dtype=np.bool_)
        customer_value = np.zeros(hardship_shape, dtype=np.float32)

        for hardship in range(2):
            for delivered in product(
                range(number_of_inventory_states),
                repeat=len(self.resource_names),
            ):
                inventory = dict(zip(self.resource_names, delivered))
                outcome = self.select_production(
                    targets,
                    inventory,
                    previous_shortfall=bool(hardship),
                )
                index = (hardship,) + delivered
                production[index] = [
                    outcome.production[name] for name in self.product_names
                ]
                ingredient_usage[index] = [
                    outcome.ingredient_usage[name] for name in self.resource_names
                ]
                leftover_inventory[index] = [
                    outcome.leftover_inventory[name] for name in self.resource_names
                ]
                served[index] = [outcome.served[name] for name in self.product_names]
                shortfall[index] = [
                    outcome.shortfall[name] for name in self.product_names
                ]
                weighted_shortfall[index] = outcome.weighted_shortfall
                complete_fulfilment[index] = outcome.complete_fulfilment
                customer_value[index] = outcome.customer_value

        arrays = (
            production,
            ingredient_usage,
            leftover_inventory,
            served,
            shortfall,
            weighted_shortfall,
            complete_fulfilment,
            customer_value,
        )
        for array in arrays:
            array.setflags(write=False)
        return SubstitutionPaymentDeductionRoundOutcomes(*arrays)


class SubstitutionPaymentDeductionOneStepSolver(PaymentDeductionOneStepSolver):
    """Payment-deduction Bellman solver using substitution-aware production."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
        price_kernel: PriceKernel | None = None,
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(environment, deduction_prices, price_kernel)
        joint = SubstitutionPaymentDeductionContingency(
            environment,
            deduction_prices,
            minimum_quantities=minimum_quantities,
        )
        self.substitution_payment_deduction = joint
        self.payment_deduction = joint
        self.substitution = joint.substitution
        self._round_outcome_cache = {}


SubstitutionPaymentDeductionBellmanSolver = (
    SubstitutionPaymentDeductionOneStepSolver
)
