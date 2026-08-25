"""Dish-substitution selector and Bellman solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from types import MappingProxyType
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bellman.solver import (
    SATISFACTION_TOLERANCE,
    OneStepSolver,
    ProductionRoundInput,
    _best_order_index,
    _select_at_best_order,
    _transition_expectation,
)
from bellman.state import WeekValueState
from contract import PaymentWeek
from environment import Environment
from environment.json_input import require_finite_number, require_integer
from transitions import DeliveryKernel, PriceKernel, SpoilageKernel


def _integer_mapping(
    value: Mapping[str, int],
    names: tuple[str, ...],
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if set(value) != set(names):
        raise ValueError(f"{label} must contain exactly {list(names)}")
    return {
        name: require_integer(value[name], name=f"{label}[{name!r}]", minimum=0)
        for name in names
    }


@dataclass(frozen=True)
class SubstitutionOutcome:
    """Selected physical production and the demand it credits."""

    production: Mapping[str, int]
    ingredient_usage: Mapping[str, int]
    leftover_inventory: Mapping[str, int]
    served: Mapping[str, int]
    shortfall: Mapping[str, int]
    exact_minimums_satisfied: bool
    substitution_points_created: int
    substitution_points_used: int
    customer_value: float

    @property
    def satisfied(self) -> bool:
        return all(quantity == 0 for quantity in self.shortfall.values())

    @property
    def complete_fulfilment(self) -> bool:
        return self.satisfied

    @property
    def remaining_shortfall(self) -> Mapping[str, int]:
        return self.shortfall


@dataclass(frozen=True, eq=False)
class SubstitutionRoundOutcomes:
    """Dense lookup indexed by the joint delivered-inventory state."""

    production: np.ndarray
    ingredient_usage: np.ndarray
    leftover_inventory: np.ndarray
    remaining_shortfall: np.ndarray
    complete_fulfilment: np.ndarray
    customer_value: np.ndarray


class SubstitutionContingency:
    """Select physical production using only public substitution points."""

    def __init__(
        self,
        environment: Environment,
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")

        self.environment = environment
        self.product_names = environment.supplier.products
        self.resource_names = environment.supplier.resources
        minimums = (
            environment.customer.minimum_products_per_production_round
            if minimum_quantities is None
            else minimum_quantities
        )
        self.minimum_quantities = MappingProxyType(
            _integer_mapping(
                minimums,
                self.product_names,
                label="minimum_quantities",
            )
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
    ) -> tuple[dict[str, int], int]:
        """Credit the most demand points."""
        point_values = self.environment.customer.substitution_rules
        best_quantities = (0,) * len(self.product_names)
        best_points = 0
        ranges = (range(capacities[dish] + 1) for dish in self.product_names)
        for quantities in product(*ranges):
            points = sum(
                quantity * point_values[dish]
                for dish, quantity in zip(self.product_names, quantities)
            )
            if points > available_points:
                continue
            if points > best_points:
                best_points = points
                best_quantities = quantities
        return dict(zip(self.product_names, best_quantities)), best_points

    def evaluate_production(
        self,
        target_quantities: Mapping[str, int],
        production: Mapping[str, int],
        available_inventory: Mapping[str, int] | None = None,
    ) -> SubstitutionOutcome:
        """Evaluate one fixed physical production plan."""
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
        served = {
            dish: exact_served[dish] + credited_above_minimum[dish]
            for dish in self.product_names
        }
        shortfall = {
            dish: targets[dish] - served[dish] for dish in self.product_names
        }
        minimums_satisfied = all(
            produced[dish] >= exact_minimums[dish] for dish in self.product_names
        )
        customer_value = float(
            sum(
                produced[dish] * self.environment.customer.private_values[dish]
                for dish in self.product_names
            )
        )
        return SubstitutionOutcome(
            production=MappingProxyType(produced),
            ingredient_usage=MappingProxyType(usage),
            leftover_inventory=MappingProxyType(leftovers),
            served=MappingProxyType(served),
            shortfall=MappingProxyType(shortfall),
            exact_minimums_satisfied=minimums_satisfied,
            substitution_points_created=points_created,
            substitution_points_used=points_used,
            customer_value=customer_value,
        )

    def select_production(
        self,
        target_quantities: Mapping[str, int],
        available_inventory: Mapping[str, int],
    ) -> SubstitutionOutcome:
        """Enumerate feasible plans and apply the required lexicographic ranking."""
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

        best_outcome: SubstitutionOutcome | None = None
        best_rank: tuple[bool, int, int] | None = None
        ranges = (
            range(maximum_production[dish] + 1) for dish in self.product_names
        )
        for quantities in product(*ranges):
            plan = dict(zip(self.product_names, quantities))
            usage = self._ingredient_usage(plan)
            if any(usage[name] > inventory[name] for name in self.resource_names):
                continue
            outcome = self.evaluate_production(targets, plan, inventory)
            credited_points = sum(
                outcome.served[dish] * point_values[dish]
                for dish in self.product_names
            )
            rank = (
                outcome.exact_minimums_satisfied,
                credited_points,
                -sum(outcome.ingredient_usage.values()),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_outcome = outcome

        assert best_outcome is not None
        return best_outcome

    def precompute_round(
        self,
        target_quantities: Mapping[str, int],
    ) -> SubstitutionRoundOutcomes:
        """Compile one selected outcome for every joint delivered inventory."""
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        number_of_inventory_states = (
            self.environment.supplier.max_inventory_per_resource + 1
        )
        inventory_shape = (number_of_inventory_states,) * len(self.resource_names)
        production = np.zeros(
            inventory_shape + (len(self.product_names),), dtype=np.int16
        )
        ingredient_usage = np.zeros(
            inventory_shape + (len(self.resource_names),), dtype=np.int16
        )
        leftover_inventory = np.zeros_like(ingredient_usage)
        remaining_shortfall = np.zeros_like(production)
        complete_fulfilment = np.zeros(inventory_shape, dtype=np.bool_)
        customer_value = np.zeros(inventory_shape, dtype=np.float32)

        for delivered in product(
            range(number_of_inventory_states),
            repeat=len(self.resource_names),
        ):
            inventory = dict(zip(self.resource_names, delivered))
            outcome = self.select_production(targets, inventory)
            production[delivered] = [
                outcome.production[name] for name in self.product_names
            ]
            ingredient_usage[delivered] = [
                outcome.ingredient_usage[name] for name in self.resource_names
            ]
            leftover_inventory[delivered] = [
                outcome.leftover_inventory[name] for name in self.resource_names
            ]
            remaining_shortfall[delivered] = [
                outcome.shortfall[name] for name in self.product_names
            ]
            complete_fulfilment[delivered] = outcome.satisfied
            customer_value[delivered] = outcome.customer_value

        arrays = (
            production,
            ingredient_usage,
            leftover_inventory,
            remaining_shortfall,
            complete_fulfilment,
            customer_value,
        )
        for array in arrays:
            array.setflags(write=False)
        return SubstitutionRoundOutcomes(*arrays)


def _spoilage_expectation(
    continuation: jax.Array,
    spoilage_tables: tuple[jax.Array, ...],
) -> jax.Array:
    """Index continuation values by leftovers before the raw-delivery step."""
    number_of_resources = len(spoilage_tables)
    expanded_tables = tuple(table[:, None, :] for table in spoilage_tables)
    expected = _transition_expectation(continuation, expanded_tables)
    order_axes = tuple(
        range(1 + number_of_resources, 1 + 2 * number_of_resources)
    )
    return jnp.squeeze(expected, axis=order_axes)


def _substitution_transition_expectation(
    continuation: jax.Array,
    delivery_tables: tuple[jax.Array, ...],
    spoilage_tables: tuple[jax.Array, ...],
    leftover_inventory: jax.Array,
    complete_fulfilment: jax.Array,
    customer_value: jax.Array,
) -> jax.Array:
    number_of_resources = len(delivery_tables)
    after_spoilage = _spoilage_expectation(continuation, spoilage_tables)
    leftover_indices = tuple(
        leftover_inventory[..., index] for index in range(number_of_resources)
    )
    selected = after_spoilage[(slice(None),) + leftover_indices + (slice(None),)]
    complete = complete_fulfilment.reshape(
        (1,) + complete_fulfilment.shape + (1,)
    )
    delivered_values = jnp.where(complete, selected, 0.0)
    delivered_values = delivered_values.at[2].add(
        jnp.where(complete_fulfilment, customer_value, 0.0)[..., None]
    )
    return _transition_expectation(delivered_values, delivery_tables)


def _substitution_bellman_step_with_policy(
    continuation: jax.Array,
    delivery_tables: tuple[jax.Array, ...],
    spoilage_tables: tuple[jax.Array, ...],
    leftover_inventory: jax.Array,
    complete_fulfilment: jax.Array,
    round_customer_value: jax.Array,
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    number_of_resources: int,
) -> tuple[jax.Array, jax.Array]:
    expected = _substitution_transition_expectation(
        continuation,
        delivery_tables,
        spoilage_tables,
        leftover_inventory,
        complete_fulfilment,
        round_customer_value,
    )
    number_of_capital_states = continuation.shape[-1]
    current_inventory_shape = expected.shape[1 : number_of_resources + 1]
    capital = jnp.arange(number_of_capital_states, dtype=cost_scenarios.dtype)

    def select_for_scenario(costs: jax.Array) -> tuple[jax.Array, jax.Array]:
        balance_after_order = capital + payment - costs[..., None]
        affordable = balance_after_order >= 0
        capital_index = jnp.clip(
            balance_after_order.astype(jnp.int32),
            0,
            number_of_capital_states - 1,
        )
        gather_index = capital_index.reshape(
            (1,) * number_of_resources + capital_index.shape
        )
        gathered = tuple(
            jnp.take_along_axis(expected[index], gather_index, axis=-1)
            for index in range(4)
        )
        action_cost = costs.reshape(
            (1,) * number_of_resources + costs.shape + (1,)
        )
        affordable = affordable.reshape(
            (1,) * number_of_resources + affordable.shape
        )
        success_probability = jnp.where(affordable, gathered[0], -1.0)
        supplier_utility = jnp.where(
            affordable,
            payment - action_cost + gathered[1],
            -jnp.inf,
        )
        customer_value = jnp.where(affordable, gathered[2], 0.0)
        total_paid = jnp.where(affordable, payment + gathered[3], 0.0)
        best = _best_order_index(
            success_probability,
            supplier_utility,
            satisfaction_cutoff,
            current_inventory_shape=current_inventory_shape,
            number_of_resources=number_of_resources,
            number_of_capital_states=number_of_capital_states,
        )
        return (
            jnp.stack(
                tuple(
                    _select_at_best_order(
                        metric,
                        best,
                        current_inventory_shape=current_inventory_shape,
                        number_of_resources=number_of_resources,
                        number_of_capital_states=number_of_capital_states,
                    )
                    for metric in (
                        success_probability,
                        supplier_utility,
                        customer_value,
                        total_paid,
                    )
                ),
                axis=0,
            ),
            best,
        )

    selected_by_scenario, policy_by_scenario = jax.vmap(
        select_for_scenario
    )(cost_scenarios)
    weights = price_probabilities.reshape(
        (price_probabilities.shape[0],)
        + (1,) * (selected_by_scenario.ndim - 1)
    )
    values = jnp.sum(selected_by_scenario * weights, axis=0)
    return values, jnp.moveaxis(policy_by_scenario, 0, -1)


_compiled_substitution_bellman_step_with_policy = jax.jit(
    _substitution_bellman_step_with_policy,
    static_argnames=("number_of_resources",),
)


class SubstitutionOneStepSolver(OneStepSolver):
    """Bellman solver with substitution-specific production transitions."""

    def __init__(
        self,
        environment: Environment,
        price_kernel: PriceKernel | None = None,
        minimum_quantities: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(environment, price_kernel)
        self.substitution = SubstitutionContingency(
            environment,
            minimum_quantities=minimum_quantities,
        )
        self.delivery = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            environment.supplier.max_inventory_per_resource,
            environment.supplier.max_order_per_resource_per_round,
        )
        self.spoilage = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            environment.supplier.max_inventory_per_resource,
        )
        self._round_outcome_cache: dict[
            tuple[int, ...], SubstitutionRoundOutcomes
        ] = {}

    def round_outcomes(
        self,
        round_input: ProductionRoundInput,
    ) -> SubstitutionRoundOutcomes:
        if not isinstance(round_input, ProductionRoundInput):
            raise TypeError("round_input must be a ProductionRoundInput")
        if set(round_input.production.product_quantities) != set(
            self.environment.supplier.products
        ):
            raise ValueError("production-week products must match the environment")
        key = tuple(
            round_input.production.product_quantities[dish]
            for dish in self.environment.supplier.products
        )
        if key not in self._round_outcome_cache:
            self._round_outcome_cache[key] = self.substitution.precompute_round(
                round_input.production.product_quantities
            )
        return self._round_outcome_cache[key]

    def solve(
        self,
        round_input: ProductionRoundInput,
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Apply one substitution-aware JIT-compiled Bellman step."""
        self._validate_continuation(continuation)
        outcomes = self.round_outcomes(round_input)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        payment = self._payment_amount(round_input.payment)
        output, _policy = _compiled_substitution_bellman_step_with_policy(
            continuation.stacked(),
            tuple(
                jnp.asarray(self.delivery.table, dtype=jnp.float32)
                for _ in self.resource_names
            ),
            tuple(
                jnp.asarray(self.spoilage.table, dtype=jnp.float32)
                for _ in self.resource_names
            ),
            jnp.asarray(outcomes.leftover_inventory, dtype=jnp.int32),
            jnp.asarray(outcomes.complete_fulfilment),
            jnp.asarray(outcomes.customer_value, dtype=jnp.float32),
            self._cost_scenarios,
            self._price_probabilities,
            jnp.asarray(payment, dtype=jnp.float32),
            jnp.asarray(
                threshold - SATISFACTION_TOLERANCE,
                dtype=jnp.float32,
            ),
            number_of_resources=len(self.resource_names),
        )
        return WeekValueState.from_stacked(output)

    def solve_rounds(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve all substitution rounds backward."""
        self._validate_continuation(continuation)
        state = continuation
        for round_input in reversed(round_inputs):
            state = self.solve(round_input, state, satisfaction_threshold)
        return state

    def solve_with_policy(
        self,
        round_input: ProductionRoundInput,
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Apply one substitution-aware step and retain selected orders."""
        self._validate_continuation(continuation)
        outcomes = self.round_outcomes(round_input)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        payment = self._payment_amount(round_input.payment)
        output, flat_policy = _compiled_substitution_bellman_step_with_policy(
            continuation.stacked(),
            tuple(
                jnp.asarray(self.delivery.table, dtype=jnp.float32)
                for _ in self.resource_names
            ),
            tuple(
                jnp.asarray(self.spoilage.table, dtype=jnp.float32)
                for _ in self.resource_names
            ),
            jnp.asarray(outcomes.leftover_inventory, dtype=jnp.int32),
            jnp.asarray(outcomes.complete_fulfilment),
            jnp.asarray(outcomes.customer_value, dtype=jnp.float32),
            self._cost_scenarios,
            self._price_probabilities,
            jnp.asarray(payment, dtype=jnp.float32),
            jnp.asarray(
                threshold - SATISFACTION_TOLERANCE,
                dtype=jnp.float32,
            ),
            number_of_resources=len(self.resource_names),
        )
        return WeekValueState.from_stacked(output), self._decode_order_policy(
            flat_policy
        )

    def solve_rounds_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve substitution rounds and retain chronological policies."""
        self._validate_continuation(continuation)
        if not round_inputs:
            shape = (0,) + self.state_shape + (
                len(self.price_kernel.scenarios),
                len(self.resource_names),
            )
            return continuation, np.empty(shape, dtype=np.int16)
        state = continuation
        reverse_policy: list[np.ndarray] = []
        for round_input in reversed(round_inputs):
            state, policy = self.solve_with_policy(
                round_input,
                state,
                satisfaction_threshold,
            )
            reverse_policy.append(policy)
        return state, np.stack(tuple(reversed(reverse_policy)), axis=0)

    def solve_rounds_from_final_payment_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve from the final payment and retain selected orders."""
        return self.solve_rounds_with_policy(
            round_inputs,
            self.terminal_state(final_payment),
            satisfaction_threshold,
        )


SubstitutionBellmanSolver = SubstitutionOneStepSolver
