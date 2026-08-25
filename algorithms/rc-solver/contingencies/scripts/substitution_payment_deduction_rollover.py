"""Joint substitution, payment-deduction, and rollover implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import ceil
from types import MappingProxyType
from typing import Mapping

import numpy as np

from environment import Environment
from environment.json_input import require_integer
from transitions import PriceKernel

from .payment_deduction import _integer_mapping
from .payment_deduction_rollover import (
    PaymentDeductionRolloverOneStepSolver,
)
from .substitution_payment_deduction import (
    SubstitutionPaymentDeductionContingency,
)


@dataclass(frozen=True)
class SubstitutionPaymentDeductionRolloverOutcome:
    """Selected production, substituted service, deduction, and deficit."""

    production: Mapping[str, int]
    ingredient_usage: Mapping[str, int]
    leftover_inventory: Mapping[str, int]
    effective_requirement: Mapping[str, int]
    exact_requirement: Mapping[str, int]
    served: Mapping[str, int]
    shortfall: Mapping[str, int]
    next_deficit: Mapping[str, int]
    weighted_shortfall: float
    complete_fulfilment: bool
    survives: bool
    exact_minimums_satisfied: bool
    substitution_points_created: int
    substitution_points_used: int
    customer_value: float

    @property
    def satisfied(self) -> bool:
        return self.complete_fulfilment

    @property
    def terminates(self) -> bool:
        return not self.survives

    @property
    def remaining_shortfall(self) -> Mapping[str, int]:
        return self.shortfall


@dataclass(frozen=True, eq=False)
class SubstitutionPaymentDeductionRolloverRoundOutcomes:
    """Dense lookup indexed by prior deficits and delivered inventory."""

    production: np.ndarray
    ingredient_usage: np.ndarray
    leftover_inventory: np.ndarray
    served: np.ndarray
    shortfall: np.ndarray
    next_deficit: np.ndarray
    weighted_shortfall: np.ndarray
    complete_fulfilment: np.ndarray
    survives: np.ndarray
    customer_value: np.ndarray


class SubstitutionPaymentDeductionRolloverContingency:
    """Substitute first, then deduct and roll one bounded residual."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
        max_deficit: int = 2,
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        joint = SubstitutionPaymentDeductionContingency(
            environment,
            deduction_prices,
            minimum_quantities,
        )
        self.substitution_payment_deduction = joint
        self.substitution = joint.substitution
        self.environment = environment
        self.product_names = environment.supplier.products
        self.resource_names = environment.supplier.resources
        self.minimum_quantities = joint.minimum_quantities
        self.deduction_prices = joint.deduction_prices
        self.max_deficit = require_integer(
            max_deficit,
            name="max_deficit",
            minimum=0,
        )

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
        exact_served: Mapping[str, int],
        effective_requirement: Mapping[str, int],
        *,
        has_previous_deficit: bool,
        allow_rollover: bool,
    ) -> tuple[dict[str, int], int]:
        """Prefer a surviving allocation, then avoid the greatest deduction."""
        point_values = self.environment.customer.substitution_rules
        best_quantities = (0,) * len(self.product_names)
        best_key: tuple[bool, bool, float, int, int] | None = None
        ranges = (range(capacities[dish] + 1) for dish in self.product_names)
        for quantities in product(*ranges):
            points = sum(
                quantity * point_values[dish]
                for dish, quantity in zip(self.product_names, quantities)
            )
            if points > available_points:
                continue
            shortfall = {
                dish: effective_requirement[dish]
                - exact_served[dish]
                - quantity
                for dish, quantity in zip(self.product_names, quantities)
            }
            complete = all(value == 0 for value in shortfall.values())
            survives = complete or (
                allow_rollover
                and not has_previous_deficit
                and all(value <= self.max_deficit for value in shortfall.values())
            )
            deduction_avoided = sum(
                quantity * self.deduction_prices[dish]
                for dish, quantity in zip(self.product_names, quantities)
            )
            key = (
                survives,
                complete,
                deduction_avoided,
                points,
                sum(quantities),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_quantities = quantities
        assert best_key is not None
        return dict(zip(self.product_names, best_quantities)), best_key[3]

    def evaluate_production(
        self,
        target_quantities: Mapping[str, int],
        previous_deficit: Mapping[str, int],
        production: Mapping[str, int],
        available_inventory: Mapping[str, int] | None = None,
        *,
        allow_rollover: bool = True,
    ) -> SubstitutionPaymentDeductionRolloverOutcome:
        """Evaluate a plan; minimums and carried deficits remain exact."""
        if not isinstance(allow_rollover, bool):
            raise TypeError("allow_rollover must be a bool")
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        deficit = _integer_mapping(
            previous_deficit,
            self.product_names,
            label="previous_deficit",
        )
        if any(value > self.max_deficit for value in deficit.values()):
            raise ValueError("previous_deficit exceeds max_deficit")
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

        base_minimum = {
            dish: min(targets[dish], self.minimum_quantities[dish])
            for dish in self.product_names
        }
        exact_requirement = {
            dish: base_minimum[dish] + deficit[dish]
            for dish in self.product_names
        }
        effective_requirement = {
            dish: targets[dish] + deficit[dish] for dish in self.product_names
        }
        exact_served = {
            dish: min(produced[dish], exact_requirement[dish])
            for dish in self.product_names
        }
        point_values = self.environment.customer.substitution_rules
        points_created = sum(
            max(produced[dish] - exact_requirement[dish], 0)
            * point_values[dish]
            for dish in self.product_names
        )
        above_minimum_demand = {
            dish: targets[dish] - base_minimum[dish]
            for dish in self.product_names
        }
        credited, points_used = self._allocate_points(
            above_minimum_demand,
            points_created,
            exact_served,
            effective_requirement,
            has_previous_deficit=any(deficit.values()),
            allow_rollover=allow_rollover,
        )
        substituted_served = {
            dish: exact_served[dish] + credited[dish]
            for dish in self.product_names
        }
        substituted_shortfall = {
            dish: effective_requirement[dish] - substituted_served[dish]
            for dish in self.product_names
        }
        if all(value == 0 for value in substituted_shortfall.values()):
            served = substituted_served
            shortfall = substituted_shortfall
            complete = True
        else:
            served = {
                dish: min(produced[dish], effective_requirement[dish])
                for dish in self.product_names
            }
            shortfall = {
                dish: effective_requirement[dish] - served[dish]
                for dish in self.product_names
            }
            points_used = 0
            complete = False
        weighted_shortfall = sum(
            shortfall[dish] * self.deduction_prices[dish]
            for dish in self.product_names
        )
        zero_deficit = {dish: 0 for dish in self.product_names}
        customer_value = sum(
            produced[dish] * self.environment.customer.private_values[dish]
            for dish in self.product_names
        )
        return SubstitutionPaymentDeductionRolloverOutcome(
            production=MappingProxyType(produced),
            ingredient_usage=MappingProxyType(usage),
            leftover_inventory=MappingProxyType(leftovers),
            effective_requirement=MappingProxyType(effective_requirement),
            exact_requirement=MappingProxyType(exact_requirement),
            served=MappingProxyType(served),
            shortfall=MappingProxyType(shortfall),
            next_deficit=MappingProxyType(zero_deficit),
            weighted_shortfall=float(weighted_shortfall),
            complete_fulfilment=complete,
            survives=complete,
            exact_minimums_satisfied=all(
                produced[dish] >= exact_requirement[dish]
                for dish in self.product_names
            ),
            substitution_points_created=points_created,
            substitution_points_used=points_used,
            customer_value=float(customer_value),
        )

    def select_production(
        self,
        target_quantities: Mapping[str, int],
        previous_deficit: Mapping[str, int],
        available_inventory: Mapping[str, int],
        *,
        allow_rollover: bool = True,
    ) -> SubstitutionPaymentDeductionRolloverOutcome:
        """Prefer survival, then least weighted loss, then least usage."""
        if not isinstance(allow_rollover, bool):
            raise TypeError("allow_rollover must be a bool")
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        deficit = _integer_mapping(
            previous_deficit,
            self.product_names,
            label="previous_deficit",
        )
        if any(value > self.max_deficit for value in deficit.values()):
            raise ValueError("previous_deficit exceeds max_deficit")
        inventory = _integer_mapping(
            available_inventory,
            self.resource_names,
            label="available_inventory",
        )
        base_minimum = {
            dish: min(targets[dish], self.minimum_quantities[dish])
            for dish in self.product_names
        }
        exact_requirement = {
            dish: base_minimum[dish] + deficit[dish]
            for dish in self.product_names
        }
        point_values = self.environment.customer.substitution_rules
        above_minimum_points = sum(
            (targets[dish] - base_minimum[dish]) * point_values[dish]
            for dish in self.product_names
        )
        maximum_production = {
            dish: exact_requirement[dish]
            + (
                ceil(above_minimum_points / point_values[dish])
                if above_minimum_points
                else 0
            )
            for dish in self.product_names
        }
        has_previous_deficit = any(deficit.values())

        best_outcome: SubstitutionPaymentDeductionRolloverOutcome | None = None
        best_rank: tuple[bool, float, int] | None = None
        ranges = (
            range(maximum_production[dish] + 1) for dish in self.product_names
        )
        for quantities in product(*ranges):
            plan = dict(zip(self.product_names, quantities))
            usage = self._ingredient_usage(plan)
            if any(usage[name] > inventory[name] for name in self.resource_names):
                continue
            outcome = self.evaluate_production(
                targets,
                deficit,
                plan,
                inventory,
                allow_rollover=allow_rollover,
            )
            valid_rollover = (
                allow_rollover
                and not has_previous_deficit
                and all(
                    value <= self.max_deficit
                    for value in outcome.shortfall.values()
                )
            )
            if not outcome.complete_fulfilment and not valid_rollover:
                continue
            rank = (
                outcome.complete_fulfilment,
                -outcome.weighted_shortfall,
                -sum(outcome.ingredient_usage.values()),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_outcome = outcome

        if best_outcome is None:
            return self.evaluate_production(
                targets,
                deficit,
                {dish: 0 for dish in self.product_names},
                inventory,
                allow_rollover=allow_rollover,
            )
        if best_outcome.complete_fulfilment:
            return best_outcome
        return replace(
            best_outcome,
            next_deficit=MappingProxyType(dict(best_outcome.shortfall)),
            survives=True,
        )

    def precompute_round(
        self,
        target_quantities: Mapping[str, int],
        *,
        allow_rollover: bool = True,
    ) -> SubstitutionPaymentDeductionRolloverRoundOutcomes:
        """Compile all prior-deficit and delivered-inventory outcomes."""
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        inventory_states = self.environment.supplier.max_inventory_per_resource + 1
        deficit_states = self.max_deficit + 1
        deficit_shape = (deficit_states,) * len(self.product_names)
        inventory_shape = (inventory_states,) * len(self.resource_names)
        state_shape = deficit_shape + inventory_shape
        production = np.zeros(
            state_shape + (len(self.product_names),), dtype=np.int16
        )
        ingredient_usage = np.zeros(
            state_shape + (len(self.resource_names),), dtype=np.int16
        )
        leftover_inventory = np.zeros_like(ingredient_usage)
        served = np.zeros_like(production)
        shortfall = np.zeros_like(production)
        next_deficit = np.zeros_like(production)
        weighted_shortfall = np.zeros(state_shape, dtype=np.float32)
        complete_fulfilment = np.zeros(state_shape, dtype=np.bool_)
        survives = np.zeros(state_shape, dtype=np.bool_)
        customer_value = np.zeros(state_shape, dtype=np.float32)

        for deficit_values in product(
            range(deficit_states),
            repeat=len(self.product_names),
        ):
            previous_deficit = dict(zip(self.product_names, deficit_values))
            for delivered in product(
                range(inventory_states),
                repeat=len(self.resource_names),
            ):
                inventory = dict(zip(self.resource_names, delivered))
                outcome = self.select_production(
                    targets,
                    previous_deficit,
                    inventory,
                    allow_rollover=allow_rollover,
                )
                index = deficit_values + delivered
                production[index] = [
                    outcome.production[name] for name in self.product_names
                ]
                ingredient_usage[index] = [
                    outcome.ingredient_usage[name] for name in self.resource_names
                ]
                leftover_inventory[index] = [
                    outcome.leftover_inventory[name] for name in self.resource_names
                ]
                served[index] = [
                    outcome.served[name] for name in self.product_names
                ]
                shortfall[index] = [
                    outcome.shortfall[name] for name in self.product_names
                ]
                next_deficit[index] = [
                    outcome.next_deficit[name] for name in self.product_names
                ]
                weighted_shortfall[index] = outcome.weighted_shortfall
                complete_fulfilment[index] = outcome.complete_fulfilment
                survives[index] = outcome.survives
                customer_value[index] = outcome.customer_value

        arrays = (
            production,
            ingredient_usage,
            leftover_inventory,
            served,
            shortfall,
            next_deficit,
            weighted_shortfall,
            complete_fulfilment,
            survives,
            customer_value,
        )
        for array in arrays:
            array.setflags(write=False)
        return SubstitutionPaymentDeductionRolloverRoundOutcomes(*arrays)


class SubstitutionPaymentDeductionRolloverOneStepSolver(
    PaymentDeductionRolloverOneStepSolver
):
    """Existing payded-rollover Bellman solver with substitution outcomes."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
        price_kernel: PriceKernel | None = None,
        max_deficit: int = 2,
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(
            environment,
            deduction_prices,
            price_kernel,
            max_deficit,
        )
        joint = SubstitutionPaymentDeductionRolloverContingency(
            environment,
            deduction_prices,
            max_deficit,
            minimum_quantities,
        )
        self.rollover = joint
        self.payment_deduction_rollover = joint
        self.substitution_payment_deduction_rollover = joint
        self.substitution_payment_deduction = (
            joint.substitution_payment_deduction
        )
        self.substitution = joint.substitution
        self._round_outcome_cache = {}


SubstitutionPaymentDeductionRolloverBellmanSolver = (
    SubstitutionPaymentDeductionRolloverOneStepSolver
)
