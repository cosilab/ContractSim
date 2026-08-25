"""Payment-deduction selector and Bellman solver."""

from __future__ import annotations

from dataclasses import dataclass
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


def _price_mapping(
    value: Mapping[str, float],
    names: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("deduction_prices must be a mapping")
    if set(value) != set(names):
        raise ValueError(
            f"deduction_prices must contain exactly {list(names)}"
        )
    return {
        name: require_finite_number(
            value[name],
            name=f"deduction_prices[{name!r}]",
            minimum=0,
        )
        for name in names
    }


@dataclass(frozen=True)
class PaymentDeductionOutcome:
    """Selected production and the shortfall it creates."""

    production: Mapping[str, int]
    ingredient_usage: Mapping[str, int]
    leftover_inventory: Mapping[str, int]
    shortfall: Mapping[str, int]
    weighted_shortfall: float
    complete_fulfilment: bool
    customer_value: float

    @property
    def satisfied(self) -> bool:
        return self.complete_fulfilment


@dataclass(frozen=True, eq=False)
class PaymentDeductionRoundOutcomes:
    """Dense outcome lookup with a leading previous-shortfall axis."""

    production: np.ndarray
    ingredient_usage: np.ndarray
    leftover_inventory: np.ndarray
    shortfall: np.ndarray
    weighted_shortfall: np.ndarray
    complete_fulfilment: np.ndarray
    customer_value: np.ndarray


class PaymentDeductionContingency:
    """Choose production by full service, weighted shortfall, then usage."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        self.environment = environment
        self.product_names = environment.supplier.products
        self.resource_names = environment.supplier.resources
        self.deduction_prices = MappingProxyType(
            _price_mapping(deduction_prices, self.product_names)
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
        target_quantities: Mapping[str, int],
        production: Mapping[str, int],
        available_inventory: Mapping[str, int] | None = None,
    ) -> PaymentDeductionOutcome:
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

        shortfall = {
            dish: max(targets[dish] - produced[dish], 0)
            for dish in self.product_names
        }
        weighted_shortfall = sum(
            shortfall[dish] * self.deduction_prices[dish]
            for dish in self.product_names
        )
        customer_value = sum(
            produced[dish] * self.environment.customer.private_values[dish]
            for dish in self.product_names
        )
        return PaymentDeductionOutcome(
            production=MappingProxyType(produced),
            ingredient_usage=MappingProxyType(usage),
            leftover_inventory=MappingProxyType(leftovers),
            shortfall=MappingProxyType(shortfall),
            weighted_shortfall=float(weighted_shortfall),
            complete_fulfilment=all(value == 0 for value in shortfall.values()),
            customer_value=float(customer_value),
        )

    def select_production(
        self,
        target_quantities: Mapping[str, int],
        available_inventory: Mapping[str, int],
        *,
        previous_shortfall: bool = False,
    ) -> PaymentDeductionOutcome:
        """Select the clause plan; a repeated shortfall invokes zero-output grim."""
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

        best_outcome: PaymentDeductionOutcome | None = None
        best_rank: tuple[bool, float, int] | None = None
        ranges = (range(targets[dish] + 1) for dish in self.product_names)
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
        if (
            previous_shortfall
            and not best_outcome.complete_fulfilment
        ):
            return self.evaluate_production(
                targets,
                {dish: 0 for dish in self.product_names},
                inventory,
            )
        return best_outcome

    def precompute_round(
        self,
        target_quantities: Mapping[str, int],
    ) -> PaymentDeductionRoundOutcomes:
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
            shortfall,
            weighted_shortfall,
            complete_fulfilment,
            customer_value,
        )
        for array in arrays:
            array.setflags(write=False)
        return PaymentDeductionRoundOutcomes(*arrays)


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


def _payment_deduction_transition_expectation(
    continuation: jax.Array,
    delivery_tables: tuple[jax.Array, ...],
    spoilage_tables: tuple[jax.Array, ...],
    leftover_inventory: jax.Array,
    weighted_shortfall: jax.Array,
    complete_fulfilment: jax.Array,
    customer_value: jax.Array,
    next_payment: jax.Array,
) -> jax.Array:
    """Apply outcome-dependent deductions before contracting raw deliveries."""
    number_of_resources = len(delivery_tables)
    number_of_capital_states = continuation.shape[-1]
    inventory_shape = complete_fulfilment.shape[1:]
    delivered_count = int(np.prod(inventory_shape))
    base_capital = jnp.arange(number_of_capital_states, dtype=jnp.int32)

    after_spoilage = jnp.stack(
        tuple(
            _spoilage_expectation(continuation[:, hardship], spoilage_tables)
            for hardship in range(2)
        ),
        axis=1,
    )

    expected_by_hardship = []
    for hardship in range(2):
        leftovers = leftover_inventory[hardship].reshape(
            (delivered_count, number_of_resources)
        )
        full = complete_fulfilment[hardship].reshape(delivered_count)
        loss = weighted_shortfall[hardship].reshape(delivered_count)
        round_value = customer_value[hardship].reshape(delivered_count)
        survives = jnp.logical_or(hardship == 0, full)
        next_hardship = jnp.logical_not(full).astype(jnp.int32)
        effective_payment = jnp.maximum(next_payment - loss, 0).astype(jnp.int32)
        next_capital = jnp.clip(
            effective_payment[:, None] + base_capital[None, :],
            0,
            number_of_capital_states - 1,
        )
        gather_indices = (
            slice(None),
            next_hardship[:, None],
            *(leftovers[:, index, None] for index in range(number_of_resources)),
            next_capital,
        )
        selected = after_spoilage[gather_indices]
        selected = jnp.where(survives[None, :, None], selected, 0.0)
        selected = selected.at[1].add(
            jnp.where(survives, effective_payment, 0)[:, None]
        )
        selected = selected.at[2].add(
            jnp.where(survives, round_value, 0.0)[:, None]
        )
        selected = selected.at[3].add(
            jnp.where(survives, effective_payment, 0)[:, None]
        )
        delivered_values = selected.reshape(
            (4,) + inventory_shape + (number_of_capital_states,)
        )
        expected_by_hardship.append(
            _transition_expectation(delivered_values, delivery_tables)
        )

    return jnp.stack(tuple(expected_by_hardship), axis=1)


def _payment_deduction_bellman_step_with_policy(
    continuation: jax.Array,
    delivery_tables: tuple[jax.Array, ...],
    spoilage_tables: tuple[jax.Array, ...],
    leftover_inventory: jax.Array,
    weighted_shortfall: jax.Array,
    complete_fulfilment: jax.Array,
    round_customer_value: jax.Array,
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    next_payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    number_of_resources: int,
) -> tuple[jax.Array, jax.Array]:
    expected = _payment_deduction_transition_expectation(
        continuation,
        delivery_tables,
        spoilage_tables,
        leftover_inventory,
        weighted_shortfall,
        complete_fulfilment,
        round_customer_value,
        next_payment,
    )
    number_of_capital_states = continuation.shape[-1]
    inventory_shape = expected.shape[2 : 2 + number_of_resources]
    current_state_shape = (2,) + inventory_shape
    number_of_state_axes = number_of_resources + 1
    capital = jnp.arange(number_of_capital_states, dtype=cost_scenarios.dtype)

    def select_for_scenario(costs: jax.Array) -> tuple[jax.Array, jax.Array]:
        balance_after_order = capital - costs[..., None]
        affordable = balance_after_order >= 0
        capital_index = jnp.clip(
            balance_after_order.astype(jnp.int32),
            0,
            number_of_capital_states - 1,
        )
        gather_index = capital_index.reshape(
            (1,) + (1,) * number_of_resources + capital_index.shape
        )
        gathered = tuple(
            jnp.take_along_axis(expected[index], gather_index, axis=-1)
            for index in range(4)
        )
        action_cost = costs.reshape(
            (1,) + (1,) * number_of_resources + costs.shape + (1,)
        )
        affordable = affordable.reshape(
            (1,) + (1,) * number_of_resources + affordable.shape
        )
        success_probability = jnp.where(affordable, gathered[0], -1.0)
        supplier_utility = jnp.where(
            affordable,
            -action_cost + gathered[1],
            -jnp.inf,
        )
        expected_customer_value = jnp.where(affordable, gathered[2], 0.0)
        expected_total_paid = jnp.where(affordable, gathered[3], 0.0)
        best = _best_order_index(
            success_probability,
            supplier_utility,
            satisfaction_cutoff,
            current_inventory_shape=current_state_shape,
            number_of_resources=number_of_state_axes,
            number_of_capital_states=number_of_capital_states,
        )
        return (
            jnp.stack(
                tuple(
                    _select_at_best_order(
                        metric,
                        best,
                        current_inventory_shape=current_state_shape,
                        number_of_resources=number_of_state_axes,
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


_compiled_payment_deduction_bellman_step_with_policy = jax.jit(
    _payment_deduction_bellman_step_with_policy,
    static_argnames=("number_of_resources",),
)


class PaymentDeductionOneStepSolver(OneStepSolver):
    """Bellman solver with one-shortfall grace and repeated-shortfall grim."""

    def __init__(
        self,
        environment: Environment,
        deduction_prices: Mapping[str, float],
        price_kernel: PriceKernel | None = None,
    ) -> None:
        super().__init__(environment, price_kernel)
        self.payment_deduction = PaymentDeductionContingency(
            environment,
            deduction_prices,
        )
        if any(
            not price.is_integer()
            for price in self.payment_deduction.deduction_prices.values()
        ):
            raise ValueError(
                "The integer capital state space requires whole deduction prices"
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
        self.state_shape = (2,) + self.state_shape
        self._round_outcome_cache: dict[
            tuple[int, ...], PaymentDeductionRoundOutcomes
        ] = {}

    def terminal_state(self) -> WeekValueState:
        """Continuation after the final adjusted payment has been processed."""
        probability = jnp.ones(self.state_shape, dtype=jnp.float32)
        zeros = jnp.zeros(self.state_shape, dtype=jnp.float32)
        return WeekValueState(probability, zeros, zeros, zeros)

    def round_outcomes(
        self,
        round_input: ProductionRoundInput,
    ) -> PaymentDeductionRoundOutcomes:
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
            self._round_outcome_cache[key] = (
                self.payment_deduction.precompute_round(
                    round_input.production.product_quantities
                )
            )
        return self._round_outcome_cache[key]

    def solve(
        self,
        round_input: ProductionRoundInput,
        next_payment: PaymentWeek,
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve one round from capital available after its preceding payment."""
        self._validate_continuation(continuation)
        if not isinstance(next_payment, PaymentWeek):
            raise TypeError("next_payment must be a PaymentWeek")
        if next_payment.week <= round_input.production.week:
            raise ValueError("next_payment must follow the production week")
        payment = self._payment_amount(next_payment)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        outcomes = self.round_outcomes(round_input)
        output, _policy = _compiled_payment_deduction_bellman_step_with_policy(
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
            jnp.asarray(outcomes.weighted_shortfall, dtype=jnp.float32),
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

    def solve_with_policy(
        self,
        round_input: ProductionRoundInput,
        next_payment: PaymentWeek,
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve one deduction-aware round and retain selected orders."""
        self._validate_continuation(continuation)
        if not isinstance(next_payment, PaymentWeek):
            raise TypeError("next_payment must be a PaymentWeek")
        if next_payment.week <= round_input.production.week:
            raise ValueError("next_payment must follow the production week")
        payment = self._payment_amount(next_payment)
        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        outcomes = self.round_outcomes(round_input)
        output, flat_policy = (
            _compiled_payment_deduction_bellman_step_with_policy(
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
                jnp.asarray(outcomes.weighted_shortfall, dtype=jnp.float32),
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
        )
        return WeekValueState.from_stacked(output), self._decode_order_policy(
            flat_policy
        )

    def _credit_initial_payment(
        self,
        state: WeekValueState,
        initial_payment: float,
    ) -> WeekValueState:
        """Shift a post-payment state back across the initial payment."""
        capital = jnp.arange(
            self.number_of_capital_states,
            dtype=jnp.int32,
        )
        paid_capital = jnp.clip(
            capital + int(initial_payment),
            0,
            self.number_of_capital_states - 1,
        )
        gather_index = paid_capital.reshape(
            (1,) * (len(self.state_shape) - 1)
            + (self.number_of_capital_states,)
        )
        gathered = tuple(
            jnp.take_along_axis(metric, gather_index, axis=-1)
            for metric in state
        )
        return WeekValueState(
            gathered[0],
            gathered[1] + initial_payment,
            gathered[2],
            gathered[3] + initial_payment,
        )

    def solve_rounds_from_final_payment(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve backward and credit the initial scheduled payment once."""
        if not round_inputs:
            raise ValueError("round_inputs must contain at least one round")
        if not isinstance(final_payment, PaymentWeek):
            raise TypeError("final_payment must be a PaymentWeek")

        following_payments = tuple(
            item.payment for item in round_inputs[1:]
        ) + (final_payment,)
        state = self.terminal_state()
        for round_input, next_payment in reversed(
            tuple(zip(round_inputs, following_payments))
        ):
            state = self.solve(
                round_input,
                next_payment,
                state,
                satisfaction_threshold,
            )

        return self._credit_initial_payment(
            state,
            self._payment_amount(round_inputs[0].payment),
        )

    def solve_rounds(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "Payment deduction needs the following payment for each round; "
            "use solve_rounds_from_final_payment"
        )

    def solve_rounds_with_policy(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "Payment deduction needs the following payment for each round; "
            "use solve_rounds_from_final_payment_with_policy"
        )

    def solve_rounds_from_final_payment_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve backward and retain chronological purchase policies."""
        if not round_inputs:
            raise ValueError("round_inputs must contain at least one round")
        if not isinstance(final_payment, PaymentWeek):
            raise TypeError("final_payment must be a PaymentWeek")

        following_payments = tuple(
            item.payment for item in round_inputs[1:]
        ) + (final_payment,)
        state = self.terminal_state()
        reverse_policy: list[np.ndarray] = []
        for round_input, next_payment in reversed(
            tuple(zip(round_inputs, following_payments))
        ):
            state, policy = self.solve_with_policy(
                round_input,
                next_payment,
                state,
                satisfaction_threshold,
            )
            reverse_policy.append(policy)

        state = self._credit_initial_payment(
            state,
            self._payment_amount(round_inputs[0].payment),
        )
        return state, np.stack(tuple(reversed(reverse_policy)), axis=0)


PaymentDeductionBellmanSolver = PaymentDeductionOneStepSolver
