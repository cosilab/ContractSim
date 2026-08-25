"""Dish-rollover selector and Bellman solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
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
class RolloverOutcome:
    """Selected production and the deficit it leaves."""

    production: Mapping[str, int]
    ingredient_usage: Mapping[str, int]
    leftover_inventory: Mapping[str, int]
    effective_requirement: Mapping[str, int]
    shortfall: Mapping[str, int]
    next_deficit: Mapping[str, int]
    total_shortfall: int
    complete_fulfilment: bool
    survives: bool
    customer_value: float

    @property
    def satisfied(self) -> bool:
        return self.complete_fulfilment

    @property
    def terminates(self) -> bool:
        return not self.survives


@dataclass(frozen=True, eq=False)
class RolloverRoundOutcomes:
    """Dense lookup indexed by prior deficits and delivered inventory."""

    production: np.ndarray
    ingredient_usage: np.ndarray
    leftover_inventory: np.ndarray
    shortfall: np.ndarray
    next_deficit: np.ndarray
    complete_fulfilment: np.ndarray
    survives: np.ndarray
    customer_value: np.ndarray


class RolloverContingency:
    """Carry one bounded dish shortfall; repeat shortfalls invoke grim."""

    def __init__(
        self,
        environment: Environment,
        max_deficit: int = 2,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        self.environment = environment
        self.product_names = environment.supplier.products
        self.resource_names = environment.supplier.resources
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

    def evaluate_production(
        self,
        effective_requirement: Mapping[str, int],
        production: Mapping[str, int],
        available_inventory: Mapping[str, int] | None = None,
    ) -> RolloverOutcome:
        requirements = _integer_mapping(
            effective_requirement,
            self.product_names,
            label="effective_requirement",
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

        shortfall = {
            dish: max(requirements[dish] - produced[dish], 0)
            for dish in self.product_names
        }
        complete = all(value == 0 for value in shortfall.values())
        zero_deficit = {dish: 0 for dish in self.product_names}
        customer_value = sum(
            produced[dish] * self.environment.customer.private_values[dish]
            for dish in self.product_names
        )
        return RolloverOutcome(
            production=MappingProxyType(produced),
            ingredient_usage=MappingProxyType(usage),
            leftover_inventory=MappingProxyType(leftovers),
            effective_requirement=MappingProxyType(requirements),
            shortfall=MappingProxyType(shortfall),
            next_deficit=MappingProxyType(zero_deficit),
            total_shortfall=sum(shortfall.values()),
            complete_fulfilment=complete,
            survives=complete,
            customer_value=float(customer_value),
        )

    def select_production(
        self,
        target_quantities: Mapping[str, int],
        previous_deficit: Mapping[str, int],
        available_inventory: Mapping[str, int],
        *,
        allow_rollover: bool = True,
    ) -> RolloverOutcome:
        """Apply complete, unweighted-shortfall, then ingredient-use ranking."""
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
        requirements = {
            dish: targets[dish] + deficit[dish] for dish in self.product_names
        }
        has_previous_deficit = any(deficit.values())

        best_outcome: RolloverOutcome | None = None
        best_rank: tuple[bool, int, int] | None = None
        ranges = (range(requirements[dish] + 1) for dish in self.product_names)
        for quantities in product(*ranges):
            plan = dict(zip(self.product_names, quantities))
            usage = self._ingredient_usage(plan)
            if any(usage[name] > inventory[name] for name in self.resource_names):
                continue
            outcome = self.evaluate_production(requirements, plan, inventory)
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
                -outcome.total_shortfall,
                -sum(outcome.ingredient_usage.values()),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_outcome = outcome

        if best_outcome is None:
            return self.evaluate_production(
                requirements,
                {dish: 0 for dish in self.product_names},
                inventory,
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
    ) -> RolloverRoundOutcomes:
        """Compile all prior-deficit and delivered-inventory outcomes."""
        targets = _integer_mapping(
            target_quantities,
            self.product_names,
            label="target_quantities",
        )
        number_of_inventory_states = (
            self.environment.supplier.max_inventory_per_resource + 1
        )
        number_of_deficit_states = self.max_deficit + 1
        deficit_shape = (number_of_deficit_states,) * len(self.product_names)
        inventory_shape = (number_of_inventory_states,) * len(self.resource_names)
        state_shape = deficit_shape + inventory_shape
        production = np.zeros(
            state_shape + (len(self.product_names),), dtype=np.int16
        )
        ingredient_usage = np.zeros(
            state_shape + (len(self.resource_names),), dtype=np.int16
        )
        leftover_inventory = np.zeros_like(ingredient_usage)
        shortfall = np.zeros_like(production)
        next_deficit = np.zeros_like(production)
        complete_fulfilment = np.zeros(state_shape, dtype=np.bool_)
        survives = np.zeros(state_shape, dtype=np.bool_)
        customer_value = np.zeros(state_shape, dtype=np.float32)

        for deficit_values in product(
            range(number_of_deficit_states),
            repeat=len(self.product_names),
        ):
            previous_deficit = dict(zip(self.product_names, deficit_values))
            for delivered in product(
                range(number_of_inventory_states),
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
                shortfall[index] = [
                    outcome.shortfall[name] for name in self.product_names
                ]
                next_deficit[index] = [
                    outcome.next_deficit[name] for name in self.product_names
                ]
                complete_fulfilment[index] = outcome.complete_fulfilment
                survives[index] = outcome.survives
                customer_value[index] = outcome.customer_value

        arrays = (
            production,
            ingredient_usage,
            leftover_inventory,
            shortfall,
            next_deficit,
            complete_fulfilment,
            survives,
            customer_value,
        )
        for array in arrays:
            array.setflags(write=False)
        return RolloverRoundOutcomes(*arrays)


def _spoilage_expectation(
    continuation: jax.Array,
    spoilage_tables: tuple[jax.Array, ...],
) -> jax.Array:
    expanded_tables = tuple(table[:, None, :] for table in spoilage_tables)
    expected = _transition_expectation(continuation, expanded_tables)
    number_of_resources = len(spoilage_tables)
    order_axes = tuple(
        range(1 + number_of_resources, 1 + 2 * number_of_resources)
    )
    return jnp.squeeze(expected, axis=order_axes)


def _rollover_spoilage_expectation(
    continuation: jax.Array,
    spoilage_tables: tuple[jax.Array, ...],
    *,
    number_of_deficit_axes: int,
) -> jax.Array:
    deficit_shape = continuation.shape[1 : 1 + number_of_deficit_axes]
    number_of_deficit_states = int(np.prod(deficit_shape))
    flattened = continuation.reshape(
        (4, number_of_deficit_states)
        + continuation.shape[1 + number_of_deficit_axes :]
    )
    return jax.vmap(
        lambda values: _spoilage_expectation(values, spoilage_tables),
        in_axes=1,
        out_axes=1,
    )(flattened)


_compiled_rollover_spoilage_expectation = jax.jit(
    _rollover_spoilage_expectation,
    static_argnames=("number_of_deficit_axes",),
)


def _rollover_deficit_step_with_policy(
    continuation_after_spoilage: jax.Array,
    delivery_tables: tuple[jax.Array, ...],
    leftover_inventory: jax.Array,
    next_deficit: jax.Array,
    survives: jax.Array,
    customer_value: jax.Array,
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    number_of_resources: int,
    number_of_deficit_states: int,
) -> tuple[jax.Array, jax.Array]:
    inventory_shape = survives.shape
    delivered_count = int(np.prod(inventory_shape))
    leftovers = leftover_inventory.reshape(
        (delivered_count, number_of_resources)
    )
    next_deficits = next_deficit.reshape(
        (delivered_count, next_deficit.shape[-1])
    )
    flat_next_deficit = jnp.zeros(delivered_count, dtype=jnp.int32)
    for index in range(next_deficits.shape[-1]):
        flat_next_deficit = (
            flat_next_deficit * number_of_deficit_states
            + next_deficits[:, index]
        )
    selected = continuation_after_spoilage[
        (
            slice(None),
            flat_next_deficit,
            *(leftovers[:, index] for index in range(number_of_resources)),
            slice(None),
        )
    ]
    survives_flat = survives.reshape(delivered_count)
    selected = jnp.where(survives_flat[None, :, None], selected, 0.0)
    selected = selected.at[2].add(
        jnp.where(
            survives_flat,
            customer_value.reshape(delivered_count),
            0.0,
        )[:, None]
    )
    delivered_values = selected.reshape(
        (4,) + inventory_shape + (continuation_after_spoilage.shape[-1],)
    )
    expected = _transition_expectation(delivered_values, delivery_tables)
    number_of_capital_states = continuation_after_spoilage.shape[-1]
    current_inventory_shape = expected.shape[1 : 1 + number_of_resources]
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
        expected_customer_value = jnp.where(affordable, gathered[2], 0.0)
        expected_total_paid = jnp.where(
            affordable,
            payment + gathered[3],
            0.0,
        )
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
                        expected_customer_value,
                        expected_total_paid,
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


_compiled_rollover_deficit_step_with_policy = jax.jit(
    _rollover_deficit_step_with_policy,
    static_argnames=("number_of_resources", "number_of_deficit_states"),
)


class RolloverOneStepSolver(OneStepSolver):
    """Bellman solver with one bounded per-dish deficit rollover."""

    def __init__(
        self,
        environment: Environment,
        price_kernel: PriceKernel | None = None,
        max_deficit: int = 2,
    ) -> None:
        super().__init__(environment, price_kernel)
        self.rollover = RolloverContingency(environment, max_deficit)
        self.delivery = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            environment.supplier.max_inventory_per_resource,
            environment.supplier.max_order_per_resource_per_round,
        )
        self.spoilage = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            environment.supplier.max_inventory_per_resource,
        )
        self.number_of_deficit_states = self.rollover.max_deficit + 1
        self.deficit_shape = (self.number_of_deficit_states,) * len(
            self.environment.supplier.products
        )
        self.base_state_shape = self.state_shape
        self.state_shape = self.deficit_shape + self.base_state_shape
        self._round_outcome_cache: dict[
            tuple[bool, int, ...], RolloverRoundOutcomes
        ] = {}

    def terminal_state(self, final_payment: PaymentWeek) -> WeekValueState:
        """Pay the post-game amount only when no dish deficit remains."""
        if not isinstance(final_payment, PaymentWeek):
            raise TypeError("final_payment must be a PaymentWeek")
        amount = self._payment_amount(final_payment)
        probability = jnp.zeros(self.state_shape, dtype=jnp.float32)
        supplier_utility = jnp.zeros(self.state_shape, dtype=jnp.float32)
        customer_value = jnp.zeros(self.state_shape, dtype=jnp.float32)
        total_paid = jnp.zeros(self.state_shape, dtype=jnp.float32)
        no_deficit = (0,) * len(self.deficit_shape) + (
            (slice(None),) * len(self.base_state_shape)
        )
        probability = probability.at[no_deficit].set(1.0)
        supplier_utility = supplier_utility.at[no_deficit].set(amount)
        total_paid = total_paid.at[no_deficit].set(amount)
        return WeekValueState(
            probability,
            supplier_utility,
            customer_value,
            total_paid,
        )

    def round_outcomes(
        self,
        round_input: ProductionRoundInput,
        *,
        allow_rollover: bool = True,
    ) -> RolloverRoundOutcomes:
        if not isinstance(round_input, ProductionRoundInput):
            raise TypeError("round_input must be a ProductionRoundInput")
        if set(round_input.production.product_quantities) != set(
            self.environment.supplier.products
        ):
            raise ValueError("production-week products must match the environment")
        key = (allow_rollover,) + tuple(
            round_input.production.product_quantities[dish]
            for dish in self.environment.supplier.products
        )
        if key not in self._round_outcome_cache:
            self._round_outcome_cache[key] = self.rollover.precompute_round(
                round_input.production.product_quantities,
                allow_rollover=allow_rollover,
            )
        return self._round_outcome_cache[key]

    def solve(
        self,
        round_input: ProductionRoundInput,
        continuation: WeekValueState,
        satisfaction_threshold: float,
        *,
        allow_rollover: bool = True,
    ) -> WeekValueState:
        """Apply one rollover-aware Bellman step."""
        self._validate_continuation(continuation)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        payment = self._payment_amount(round_input.payment)
        outcomes = self.round_outcomes(
            round_input,
            allow_rollover=allow_rollover,
        )
        delivery_tables = tuple(
            jnp.asarray(self.delivery.table, dtype=jnp.float32)
            for _ in self.resource_names
        )
        spoilage_tables = tuple(
            jnp.asarray(self.spoilage.table, dtype=jnp.float32)
            for _ in self.resource_names
        )
        after_spoilage = _compiled_rollover_spoilage_expectation(
            continuation.stacked(),
            spoilage_tables,
            number_of_deficit_axes=len(self.deficit_shape),
        )
        selected_by_deficit = []
        for deficit in product(
            range(self.number_of_deficit_states),
            repeat=len(self.deficit_shape),
        ):
            selected, _policy = _compiled_rollover_deficit_step_with_policy(
                after_spoilage,
                delivery_tables,
                jnp.asarray(
                    outcomes.leftover_inventory[deficit],
                    dtype=jnp.int32,
                ),
                jnp.asarray(
                    outcomes.next_deficit[deficit],
                    dtype=jnp.int32,
                ),
                jnp.asarray(outcomes.survives[deficit]),
                jnp.asarray(
                    outcomes.customer_value[deficit],
                    dtype=jnp.float32,
                ),
                self._cost_scenarios,
                self._price_probabilities,
                jnp.asarray(payment, dtype=jnp.float32),
                jnp.asarray(
                    threshold - SATISFACTION_TOLERANCE,
                    dtype=jnp.float32,
                ),
                number_of_resources=len(self.resource_names),
                number_of_deficit_states=self.number_of_deficit_states,
            )
            selected_by_deficit.append(selected)
        output = jnp.stack(tuple(selected_by_deficit), axis=1).reshape(
            (4,) + self.state_shape
        )
        return WeekValueState.from_stacked(output)

    def solve_with_policy(
        self,
        round_input: ProductionRoundInput,
        continuation: WeekValueState,
        satisfaction_threshold: float,
        *,
        allow_rollover: bool = True,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Apply one rollover-aware step and retain selected orders."""
        self._validate_continuation(continuation)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        payment = self._payment_amount(round_input.payment)
        outcomes = self.round_outcomes(
            round_input,
            allow_rollover=allow_rollover,
        )
        delivery_tables = tuple(
            jnp.asarray(self.delivery.table, dtype=jnp.float32)
            for _ in self.resource_names
        )
        spoilage_tables = tuple(
            jnp.asarray(self.spoilage.table, dtype=jnp.float32)
            for _ in self.resource_names
        )
        after_spoilage = _compiled_rollover_spoilage_expectation(
            continuation.stacked(),
            spoilage_tables,
            number_of_deficit_axes=len(self.deficit_shape),
        )
        selected_by_deficit = []
        policy_by_deficit = []
        for deficit in product(
            range(self.number_of_deficit_states),
            repeat=len(self.deficit_shape),
        ):
            selected, flat_policy = _compiled_rollover_deficit_step_with_policy(
                after_spoilage,
                delivery_tables,
                jnp.asarray(
                    outcomes.leftover_inventory[deficit],
                    dtype=jnp.int32,
                ),
                jnp.asarray(
                    outcomes.next_deficit[deficit],
                    dtype=jnp.int32,
                ),
                jnp.asarray(outcomes.survives[deficit]),
                jnp.asarray(
                    outcomes.customer_value[deficit],
                    dtype=jnp.float32,
                ),
                self._cost_scenarios,
                self._price_probabilities,
                jnp.asarray(payment, dtype=jnp.float32),
                jnp.asarray(
                    threshold - SATISFACTION_TOLERANCE,
                    dtype=jnp.float32,
                ),
                number_of_resources=len(self.resource_names),
                number_of_deficit_states=self.number_of_deficit_states,
            )
            selected_by_deficit.append(selected)
            policy_by_deficit.append(flat_policy)
        output = jnp.stack(tuple(selected_by_deficit), axis=1).reshape(
            (4,) + self.state_shape
        )
        flat_policy = jnp.stack(tuple(policy_by_deficit), axis=0).reshape(
            self.state_shape + (len(self.price_kernel.scenarios),)
        )
        return WeekValueState.from_stacked(output), self._decode_order_policy(
            flat_policy
        )

    def solve_rounds(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve rounds that may all roll into the supplied continuation."""
        self._validate_continuation(continuation)
        state = continuation
        for round_input in reversed(round_inputs):
            state = self.solve(
                round_input,
                state,
                satisfaction_threshold,
                allow_rollover=True,
            )
        return state

    def solve_rounds_from_final_payment(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve backward; the last production round cannot create a deficit."""
        if not round_inputs:
            return self.terminal_state(final_payment)
        state = self.terminal_state(final_payment)
        final_index = len(round_inputs) - 1
        for index in reversed(range(len(round_inputs))):
            state = self.solve(
                round_inputs[index],
                state,
                satisfaction_threshold,
                allow_rollover=index < final_index,
            )
        return state

    def solve_rounds_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve rollover rounds and retain chronological policies."""
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
                allow_rollover=True,
            )
            reverse_policy.append(policy)
        return state, np.stack(tuple(reversed(reverse_policy)), axis=0)

    def solve_rounds_from_final_payment_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve backward and retain chronological purchase policies."""
        if not round_inputs:
            terminal = self.terminal_state(final_payment)
            shape = (0,) + self.state_shape + (
                len(self.price_kernel.scenarios),
                len(self.resource_names),
            )
            return terminal, np.empty(shape, dtype=np.int16)
        state = self.terminal_state(final_payment)
        reverse_policy: list[np.ndarray] = []
        final_index = len(round_inputs) - 1
        for index in reversed(range(len(round_inputs))):
            state, policy = self.solve_with_policy(
                round_inputs[index],
                state,
                satisfaction_threshold,
                allow_rollover=index < final_index,
            )
            reverse_policy.append(policy)
        return state, np.stack(tuple(reversed(reverse_policy)), axis=0)


RolloverBellmanSolver = RolloverOneStepSolver
