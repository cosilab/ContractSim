"""JAX Bellman operations for exact-delivery grim-trigger contracts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from contract import PaymentWeek, ProductionWeek
from environment import Environment
from environment.json_input import require_finite_number
from transitions import PriceKernel, RoundTransitionKernels

from .contractions import sequential_contraction_steps
from .state import WeekValueState

SATISFACTION_TOLERANCE = 1e-4

JAX_CACHE_DIRECTORY = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax"),
)
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIRECTORY)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@dataclass(frozen=True, eq=False)
class ProductionRoundInput:
    """One production week paired with its preceding payment and transitions."""

    production: ProductionWeek
    payment: PaymentWeek
    transitions: RoundTransitionKernels

    def __post_init__(self) -> None:
        if not isinstance(self.production, ProductionWeek):
            raise TypeError("production must be a ProductionWeek")
        if not isinstance(self.payment, PaymentWeek):
            raise TypeError("payment must be a PaymentWeek")
        if not isinstance(self.transitions, RoundTransitionKernels):
            raise TypeError("transitions must be RoundTransitionKernels")
        if self.transitions.week != self.production.week:
            raise ValueError("production and transition weeks must match")
        if self.payment.week >= self.production.week:
            raise ValueError("the production-round payment must precede production")


def _transition_expectation(
    continuation: jax.Array,
    transition_tables: tuple[jax.Array, ...],
) -> jax.Array:
    """Contract resources sequentially in ContractBench resource order."""
    number_of_resources = len(transition_tables)
    if continuation.ndim != number_of_resources + 2:
        raise ValueError(
            "continuation must have metric, resource-inventory, and capital axes"
        )

    value = continuation
    for table, (table_labels, value_labels, output_labels) in zip(
        transition_tables,
        sequential_contraction_steps(number_of_resources),
    ):
        value = jnp.einsum(
            f"{table_labels},{value_labels}->{output_labels}",
            table,
            value,
            optimize=False,
        )
    return value


def _best_order_index(
    success_probability: jax.Array,
    supplier_utility: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    current_inventory_shape: tuple[int, ...],
    number_of_resources: int,
    number_of_capital_states: int,
) -> jax.Array:
    """Apply the original ContractBench float32 action-scoring rule."""
    flattened_probability = success_probability.reshape(
        current_inventory_shape + (-1, number_of_capital_states)
    )
    flattened_supplier_utility = supplier_utility.reshape(
        current_inventory_shape + (-1, number_of_capital_states)
    )
    action_axis = number_of_resources
    good = flattened_probability >= satisfaction_cutoff
    score = jnp.where(
        good,
        1e6 + flattened_supplier_utility,
        flattened_probability,
    )
    return jnp.argmax(score, axis=action_axis)


def _select_at_best_order(
    metric: jax.Array,
    best_order: jax.Array,
    *,
    current_inventory_shape: tuple[int, ...],
    number_of_resources: int,
    number_of_capital_states: int,
) -> jax.Array:
    """Gather one action metric at a selected flattened order index."""
    flattened_metric = metric.reshape(
        current_inventory_shape + (-1, number_of_capital_states)
    )
    best_index = jnp.expand_dims(best_order, axis=number_of_resources)
    selected = jnp.take_along_axis(
        flattened_metric,
        best_index,
        axis=number_of_resources,
    )
    return jnp.squeeze(selected, axis=number_of_resources)


def _bellman_step(
    continuation: jax.Array,
    transition_tables: tuple[jax.Array, ...],
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    customer_value: jax.Array,
    *,
    number_of_resources: int,
) -> jax.Array:
    """Pure JAX Bellman step shared by one-round solves and ``lax.scan``."""
    values, _policy = _bellman_step_with_policy(
        continuation,
        transition_tables,
        cost_scenarios,
        price_probabilities,
        payment,
        satisfaction_cutoff,
        customer_value,
        number_of_resources=number_of_resources,
    )
    return values


def _bellman_step_with_policy(
    continuation: jax.Array,
    transition_tables: tuple[jax.Array, ...],
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    customer_value: jax.Array,
    *,
    number_of_resources: int,
) -> tuple[jax.Array, jax.Array]:
    """Return a Bellman step and its selected flat order indices."""
    expected = _transition_expectation(
        continuation,
        transition_tables,
    )
    number_of_capital_states = continuation.shape[-1]
    current_inventory_shape = expected.shape[
        1 : number_of_resources + 1
    ]
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
        gathered_probability = jnp.take_along_axis(
            expected[0],
            gather_index,
            axis=-1,
        )
        gathered_supplier_utility = jnp.take_along_axis(
            expected[1],
            gather_index,
            axis=-1,
        )
        gathered_customer_value = jnp.take_along_axis(
            expected[2],
            gather_index,
            axis=-1,
        )
        gathered_total_paid = jnp.take_along_axis(
            expected[3],
            gather_index,
            axis=-1,
        )

        action_cost = costs.reshape(
            (1,) * number_of_resources + costs.shape + (1,)
        )
        affordable = affordable.reshape(
            (1,) * number_of_resources + affordable.shape
        )
        success_probability = jnp.where(
            affordable,
            gathered_probability,
            -1.0,
        )
        supplier_utility = jnp.where(
            affordable,
            payment - action_cost + gathered_supplier_utility,
            -jnp.inf,
        )
        expected_customer_value = jnp.where(
            affordable,
            gathered_customer_value,
            0.0,
        )
        expected_total_paid = jnp.where(
            affordable,
            payment + gathered_total_paid,
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
        selected_probability = _select_at_best_order(
            success_probability,
            best,
            current_inventory_shape=current_inventory_shape,
            number_of_resources=number_of_resources,
            number_of_capital_states=number_of_capital_states,
        )
        selected_supplier_utility = _select_at_best_order(
            supplier_utility,
            best,
            current_inventory_shape=current_inventory_shape,
            number_of_resources=number_of_resources,
            number_of_capital_states=number_of_capital_states,
        )
        selected_customer_value = _select_at_best_order(
            expected_customer_value,
            best,
            current_inventory_shape=current_inventory_shape,
            number_of_resources=number_of_resources,
            number_of_capital_states=number_of_capital_states,
        )
        selected_total_paid = _select_at_best_order(
            expected_total_paid,
            best,
            current_inventory_shape=current_inventory_shape,
            number_of_resources=number_of_resources,
            number_of_capital_states=number_of_capital_states,
        )

        return (
            jnp.stack(
                (
                    selected_probability,
                    selected_supplier_utility,
                    selected_customer_value
                    + selected_probability * customer_value,
                    selected_total_paid,
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
    policy = jnp.moveaxis(policy_by_scenario, 0, -1)
    return values, policy


@partial(
    jax.jit,
    static_argnames=("number_of_resources",),
)
def _compiled_bellman_step(
    continuation: jax.Array,
    transition_tables: tuple[jax.Array, ...],
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    payment: jax.Array,
    satisfaction_cutoff: jax.Array,
    customer_value: jax.Array,
    *,
    number_of_resources: int,
) -> jax.Array:
    return _bellman_step(
        continuation,
        transition_tables,
        cost_scenarios,
        price_probabilities,
        payment,
        satisfaction_cutoff,
        customer_value,
        number_of_resources=number_of_resources,
    )


@partial(
    jax.jit,
    static_argnames=("number_of_resources",),
)
def _compiled_backward_scan(
    continuation: jax.Array,
    payments: jax.Array,
    customer_values: jax.Array,
    transition_tables_by_resource: tuple[jax.Array, ...],
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    number_of_resources: int,
) -> jax.Array:
    def scan_step(
        carry: jax.Array,
        round_data: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, None]:
        payment, customer_value, transition_tables = round_data
        next_carry = _bellman_step(
            carry,
            transition_tables,
            cost_scenarios,
            price_probabilities,
            payment,
            satisfaction_cutoff,
            customer_value,
            number_of_resources=number_of_resources,
        )
        return next_carry, None

    final_state, _ = jax.lax.scan(
        scan_step,
        continuation,
        (payments, customer_values, transition_tables_by_resource),
    )
    return final_state


@partial(
    jax.jit,
    static_argnames=("number_of_resources",),
)
def _compiled_backward_scan_with_policy(
    continuation: jax.Array,
    payments: jax.Array,
    customer_values: jax.Array,
    transition_tables_by_resource: tuple[jax.Array, ...],
    cost_scenarios: jax.Array,
    price_probabilities: jax.Array,
    satisfaction_cutoff: jax.Array,
    *,
    number_of_resources: int,
) -> tuple[jax.Array, jax.Array]:
    """Run the backward scan and retain each round's selected actions."""
    def scan_step(
        carry: jax.Array,
        round_data: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    ) -> tuple[jax.Array, jax.Array]:
        payment, customer_value, transition_tables = round_data
        return _bellman_step_with_policy(
            carry,
            transition_tables,
            cost_scenarios,
            price_probabilities,
            payment,
            satisfaction_cutoff,
            customer_value,
            number_of_resources=number_of_resources,
        )

    final_state, reverse_policy = jax.lax.scan(
        scan_step,
        continuation,
        (payments, customer_values, transition_tables_by_resource),
    )
    return final_state, jnp.flip(reverse_policy, axis=0)


class OneStepSolver:
    """Environment-bound JAX Bellman operator for exact production rounds."""

    def __init__(
        self,
        environment: Environment,
        price_kernel: PriceKernel | None = None,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        if price_kernel is None:
            price_kernel = PriceKernel(
                environment.probability_dist.price_distribution
            )
        if not isinstance(price_kernel, PriceKernel):
            raise TypeError("price_kernel must be a PriceKernel")

        resources = environment.supplier.resources
        if price_kernel.resource_names != resources:
            raise ValueError("price-kernel resources must match environment resources")
        if not np.allclose(price_kernel.scenarios, np.round(price_kernel.scenarios)):
            raise ValueError(
                "The integer capital state space requires integer resource prices"
            )

        self.environment = environment
        self.price_kernel = price_kernel
        self.resource_names = resources
        self.number_of_inventory_states = (
            environment.supplier.max_inventory_per_resource + 1
        )
        self.number_of_order_states = (
            environment.supplier.max_order_per_resource_per_round + 1
        )
        self.number_of_capital_states = (
            environment.supplier.number_of_capital_states
        )
        self.state_shape = (
            (self.number_of_inventory_states,) * len(self.resource_names)
            + (self.number_of_capital_states,)
        )
        self._cost_scenarios = self._compile_cost_scenarios()
        self._price_probabilities = jax.device_put(
            jnp.asarray(price_kernel.probabilities, dtype=jnp.float32)
        )

    def _compile_cost_scenarios(self) -> jax.Array:
        number_of_resources = len(self.resource_names)
        number_of_scenarios = len(self.price_kernel.scenarios)
        order_shape = (self.number_of_order_states,) * number_of_resources
        costs = np.zeros((number_of_scenarios,) + order_shape, dtype=np.float32)
        orders = np.arange(self.number_of_order_states, dtype=np.float32)

        for resource_index in range(number_of_resources):
            price_shape = (number_of_scenarios,) + (1,) * number_of_resources
            prices = self.price_kernel.scenarios[:, resource_index].reshape(
                price_shape
            )
            order_axis = (
                (1,)
                + (1,) * resource_index
                + (self.number_of_order_states,)
                + (1,) * (number_of_resources - resource_index - 1)
            )
            costs += prices * orders.reshape(order_axis)
        return jax.device_put(jnp.asarray(costs))

    def _decode_order_policy(self, flat_policy: jax.Array) -> np.ndarray:
        """Decode flattened order indices into one quantity per resource."""
        remaining = np.asarray(flat_policy, dtype=np.int32).copy()
        policy = np.empty(
            remaining.shape + (len(self.resource_names),),
            dtype=np.int16,
        )
        for resource_index in reversed(range(len(self.resource_names))):
            policy[..., resource_index] = (
                remaining % self.number_of_order_states
            )
            remaining //= self.number_of_order_states
        return policy

    @staticmethod
    def _payment_amount(payment: PaymentWeek) -> float:
        amount = require_finite_number(
            payment.amount,
            name=f"payment for week {payment.week}",
            minimum=0,
        )
        if not amount.is_integer():
            raise ValueError("The integer capital state space requires whole payments")
        return amount

    def terminal_state(self, final_payment: PaymentWeek) -> WeekValueState:
        """Create the continuation state after receiving the final payment."""
        if not isinstance(final_payment, PaymentWeek):
            raise TypeError("final_payment must be a PaymentWeek")
        amount = self._payment_amount(final_payment)
        satisfaction_probability = jnp.ones(
            self.state_shape,
            dtype=jnp.float32,
        )
        supplier_utility = jnp.full(
            self.state_shape,
            amount,
            dtype=jnp.float32,
        )
        customer_value = jnp.zeros(
            self.state_shape,
            dtype=jnp.float32,
        )
        total_paid = jnp.full(
            self.state_shape,
            amount,
            dtype=jnp.float32,
        )
        return WeekValueState(
            satisfaction_probability,
            supplier_utility,
            customer_value,
            total_paid,
        )

    def _validate_continuation(self, continuation: WeekValueState) -> None:
        if not isinstance(continuation, WeekValueState):
            raise TypeError("continuation must be a WeekValueState")
        if continuation.shape != self.state_shape:
            raise ValueError(
                f"continuation shape must be {self.state_shape}; "
                f"got {continuation.shape}"
            )

    def _transition_tables(
        self,
        round_input: ProductionRoundInput,
    ) -> tuple[jax.Array, ...]:
        if set(round_input.transitions.kernels_by_resource) != set(
            self.resource_names
        ):
            raise ValueError("round transition resources must match the environment")

        expected_shape = (
            self.number_of_inventory_states,
            self.number_of_order_states,
            self.number_of_inventory_states,
        )
        tables: list[jax.Array] = []
        for resource in self.resource_names:
            table = round_input.transitions.kernels_by_resource[
                resource
            ].successful_table
            if table.shape != expected_shape:
                raise ValueError(
                    f"transition table for {resource!r} has shape {table.shape}; "
                    f"expected {expected_shape}"
                )
            tables.append(jnp.asarray(table, dtype=jnp.float32))
        return tuple(tables)

    def _round_values(
        self,
        round_input: ProductionRoundInput,
        satisfaction_threshold: float,
    ) -> tuple[float, float, float]:
        if not isinstance(round_input, ProductionRoundInput):
            raise TypeError("round_input must be a ProductionRoundInput")
        products = set(self.environment.supplier.products)
        if set(round_input.production.product_quantities) != products:
            raise ValueError("production-week products must match the environment")

        threshold = require_finite_number(
            satisfaction_threshold,
            name="satisfaction_threshold",
            minimum=0,
            maximum=1,
        )
        payment = self._payment_amount(round_input.payment)
        customer_value = float(
            sum(
                round_input.production.product_quantities[product]
                * self.environment.customer.private_values[product]
                for product in self.environment.supplier.products
            )
        )
        satisfaction_cutoff = threshold - SATISFACTION_TOLERANCE
        return payment, satisfaction_cutoff, customer_value

    def solve(
        self,
        round_input: ProductionRoundInput,
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Apply one JIT-compiled Bellman step for a production week."""
        self._validate_continuation(continuation)
        payment, satisfaction_cutoff, customer_value = self._round_values(
            round_input,
            satisfaction_threshold,
        )
        output = _compiled_bellman_step(
            continuation.stacked(),
            self._transition_tables(round_input),
            self._cost_scenarios,
            self._price_probabilities,
            jnp.asarray(payment, dtype=jnp.float32),
            jnp.asarray(satisfaction_cutoff, dtype=jnp.float32),
            jnp.asarray(customer_value, dtype=jnp.float32),
            number_of_resources=len(self.resource_names),
        )
        return WeekValueState.from_stacked(output)

    def _scan_inputs(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        satisfaction_threshold: float,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...], float]:
        """Validate and stack reverse-calendar inputs for a compiled scan."""
        payments: list[float] = []
        customer_values: list[float] = []
        tables_by_round: list[tuple[jax.Array, ...]] = []
        satisfaction_cutoff: float | None = None

        for round_input in reversed(round_inputs):
            payment, round_cutoff, customer_value = self._round_values(
                round_input,
                satisfaction_threshold,
            )
            payments.append(payment)
            customer_values.append(customer_value)
            tables_by_round.append(self._transition_tables(round_input))
            satisfaction_cutoff = round_cutoff

        assert satisfaction_cutoff is not None
        transition_tables_by_resource = tuple(
            jnp.stack(
                [tables[resource_index] for tables in tables_by_round],
                axis=0,
            )
            for resource_index in range(len(self.resource_names))
        )
        return (
            jnp.asarray(payments, dtype=jnp.float32),
            jnp.asarray(customer_values, dtype=jnp.float32),
            transition_tables_by_resource,
            satisfaction_cutoff,
        )

    def solve_rounds(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Apply all rounds backward in one JIT-compiled ``lax.scan``."""
        self._validate_continuation(continuation)
        if not round_inputs:
            return continuation

        (
            payments,
            customer_values,
            transition_tables_by_resource,
            satisfaction_cutoff,
        ) = self._scan_inputs(
            round_inputs,
            satisfaction_threshold,
        )
        output = _compiled_backward_scan(
            continuation.stacked(),
            payments,
            customer_values,
            transition_tables_by_resource,
            self._cost_scenarios,
            self._price_probabilities,
            jnp.asarray(satisfaction_cutoff, dtype=jnp.float32),
            number_of_resources=len(self.resource_names),
        )
        return WeekValueState.from_stacked(output)

    def solve_rounds_from_final_payment(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> WeekValueState:
        """Solve rounds backward from the contract's final payment."""
        return self.solve_rounds(
            round_inputs,
            self.terminal_state(final_payment),
            satisfaction_threshold,
        )

    def solve_rounds_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        continuation: WeekValueState,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve all rounds and return decoded order quantities by state."""
        self._validate_continuation(continuation)
        if not round_inputs:
            empty_shape = (0,) + self.state_shape + (
                len(self.price_kernel.scenarios),
                len(self.resource_names),
            )
            return continuation, np.empty(empty_shape, dtype=np.int16)

        (
            payments,
            customer_values,
            transition_tables_by_resource,
            satisfaction_cutoff,
        ) = self._scan_inputs(round_inputs, satisfaction_threshold)
        output, flat_policy = _compiled_backward_scan_with_policy(
            continuation.stacked(),
            payments,
            customer_values,
            transition_tables_by_resource,
            self._cost_scenarios,
            self._price_probabilities,
            jnp.asarray(satisfaction_cutoff, dtype=jnp.float32),
            number_of_resources=len(self.resource_names),
        )
        return WeekValueState.from_stacked(output), self._decode_order_policy(
            flat_policy
        )

    def solve_rounds_from_final_payment_with_policy(
        self,
        round_inputs: tuple[ProductionRoundInput, ...],
        final_payment: PaymentWeek,
        satisfaction_threshold: float,
    ) -> tuple[WeekValueState, np.ndarray]:
        """Solve from the final payment and retain the order policy."""
        return self.solve_rounds_with_policy(
            round_inputs,
            self.terminal_state(final_payment),
            satisfaction_threshold,
        )
