"""Exact absorbing-dead occupancy completeness for modular solver policies."""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import numpy as np


METRIC_VERSION = 1


def shift_capital(
    mass: np.ndarray,
    delta: int | np.ndarray,
    number_of_states: int,
) -> np.ndarray:
    """Shift the final capital axis; unaffordable mass is dropped."""
    values = np.asarray(mass, dtype=np.float64)
    flat = values.reshape(-1, number_of_states)
    shifts = np.broadcast_to(np.asarray(delta), values.shape[:-1]).reshape(-1)
    output = np.zeros_like(flat)
    sources = np.arange(number_of_states)
    for row, amount in enumerate(shifts):
        destinations = sources + int(amount)
        keep = destinations >= 0
        np.add.at(
            output[row],
            np.minimum(destinations[keep], number_of_states - 1),
            flat[row, keep],
        )
    return output.reshape(values.shape)


def apply_kernels(mass: np.ndarray, kernels: Sequence[np.ndarray]) -> np.ndarray:
    """Push independent inventory axes through current-to-next kernels."""
    output = np.asarray(mass, dtype=np.float64)
    for axis, kernel in enumerate(kernels):
        output = np.tensordot(np.asarray(kernel), output, axes=([0], [axis]))
        output = np.moveaxis(output, 0, axis)
    return output


def order_codes(orders: np.ndarray, number_of_orders: int) -> np.ndarray:
    codes = np.zeros(orders.shape[:-1], dtype=np.int64)
    for resource in range(orders.shape[-1]):
        codes = codes * number_of_orders + orders[..., resource]
    return codes


def decode_order(
    code: int,
    number_of_resources: int,
    number_of_orders: int,
) -> tuple[int, ...]:
    orders = [0] * number_of_resources
    for resource in reversed(range(number_of_resources)):
        orders[resource] = code % number_of_orders
        code //= number_of_orders
    return tuple(orders)


def delivery_step(
    mass: np.ndarray,
    policy: np.ndarray,
    scenario_index: int,
    prices: np.ndarray,
    solver: Any,
    payment: int,
    delivery_tables: Sequence[np.ndarray],
) -> np.ndarray:
    """Apply policy orders, affordability, and stochastic delivery."""
    output = np.zeros_like(mass, dtype=np.float64)
    selected_policy = policy[..., scenario_index, :]
    codes = order_codes(selected_policy, solver.number_of_order_states)
    positive = mass > 0
    for code in np.unique(codes[positive]):
        orders = decode_order(
            int(code), len(solver.resource_names), solver.number_of_order_states
        )
        selected = np.where((codes == code) & positive, mass, 0.0)
        cost = int(sum(price * order for price, order in zip(prices, orders)))
        selected = shift_capital(
            selected,
            payment - cost,
            solver.number_of_capital_states,
        )
        output += apply_kernels(
            selected,
            [table[:, order, :] for table, order in zip(delivery_tables, orders)],
        )
    return output


def production_step(
    delivered: np.ndarray,
    solver: Any,
    outcomes: Any,
    contingency_state: tuple[int, ...],
    contingency_shape: tuple[int, ...],
    specification: Any,
    next_payment: int,
) -> np.ndarray:
    """Apply production, contingency state, adjusted payment, and spoilage."""
    prefix = contingency_state
    leftovers = np.asarray(outcomes.leftover_inventory[prefix])
    inventory_shape = delivered.shape[:-1]
    if specification.rollover:
        survives = np.asarray(outcomes.survives[prefix], dtype=np.bool_)
        next_states = np.asarray(outcomes.next_deficit[prefix])
    elif specification.payment_deduction:
        complete = np.asarray(outcomes.complete_fulfilment[prefix], dtype=np.bool_)
        survives = np.logical_or(contingency_state[0] == 0, complete)
        next_states = np.logical_not(complete).astype(np.int64)[..., None]
    else:
        survives = np.asarray(outcomes.complete_fulfilment, dtype=np.bool_)
        next_states = np.zeros(inventory_shape + (0,), dtype=np.int64)

    if specification.payment_deduction:
        shortfall = np.rint(outcomes.weighted_shortfall[prefix]).astype(np.int64)
        payment = np.maximum(next_payment - shortfall, 0)
    else:
        payment = np.zeros(inventory_shape, dtype=np.int64)

    output = np.zeros(contingency_shape + delivered.shape, dtype=np.float64)
    live = survives & (delivered.sum(axis=-1) > 0)
    for raw_inventory in np.argwhere(live):
        inventory = tuple(int(value) for value in raw_inventory)
        rows = shift_capital(
            delivered[inventory][None, :],
            int(payment[inventory]),
            solver.number_of_capital_states,
        )[0]
        next_contingency = tuple(int(value) for value in next_states[inventory])
        leftover = tuple(int(value) for value in leftovers[inventory])
        output[next_contingency + leftover] += rows

    spoilage = np.asarray(solver.spoilage.table)
    states = product(*(range(size) for size in contingency_shape))
    if contingency_shape:
        for state in states:
            output[state] = apply_kernels(
                output[state], [spoilage] * len(solver.resource_names)
            )
    else:
        output = apply_kernels(
            output, [spoilage] * len(solver.resource_names)
        )
    return output


def exact_round(
    mass: np.ndarray,
    policy: np.ndarray,
    solver: Any,
    round_input: Any,
) -> np.ndarray:
    """Push a base grim-trigger round through its composed success kernels."""
    output = np.zeros_like(mass, dtype=np.float64)
    tables = [
        round_input.transitions.kernels_by_resource[name].successful_table
        for name in solver.resource_names
    ]
    probabilities = np.asarray(solver.price_kernel.probabilities)
    for scenario, (prices, probability) in enumerate(
        zip(solver.price_kernel.scenarios, probabilities)
    ):
        output += probability * delivery_step(
            mass,
            policy,
            scenario,
            prices,
            solver,
            int(round_input.payment.amount),
            tables,
        )
    return output


def occupancy_completeness(
    policies: np.ndarray,
    *,
    solver: Any,
    round_inputs: Sequence[Any],
    final_payment: Any,
    specification: Any,
    starting_capital: int,
    p_sat: float | None = None,
) -> dict[str, Any]:
    """Return expected fraction of rounds spent in compliant operation."""
    rounds = tuple(round_inputs)
    contingency_shape = (
        tuple(solver.deficit_shape)
        if specification.rollover
        else ((2,) if specification.payment_deduction else ())
    )
    inventory_shape = (solver.number_of_inventory_states,) * len(
        solver.resource_names
    )
    state_shape = contingency_shape + inventory_shape + (
        solver.number_of_capital_states,
    )
    expected_policy_shape = (
        (len(rounds),)
        + state_shape
        + (len(solver.price_kernel.scenarios), len(solver.resource_names))
    )
    policy = np.asarray(policies)
    if not rounds or policy.shape != expected_policy_shape:
        raise ValueError(
            f"expected non-empty rounds and policy shape {expected_policy_shape}; "
            f"got {policy.shape}"
        )

    initial_capital = int(starting_capital)
    if specification.payment_deduction:
        initial_capital += int(rounds[0].payment.amount)
    initial_capital = min(initial_capital, solver.number_of_capital_states - 1)
    mass = np.zeros(state_shape, dtype=np.float64)
    initial = (0,) * (len(state_shape) - 1) + (initial_capital,)
    mass[initial] = 1.0
    survival_curve: list[float] = []
    base_mode = not (
        specification.substitution
        or specification.payment_deduction
        or specification.rollover
    )
    prices = np.asarray(solver.price_kernel.scenarios)
    price_probabilities = np.asarray(solver.price_kernel.probabilities)
    delivery_tables = (
        [np.asarray(solver.delivery.table)] * len(solver.resource_names)
        if not base_mode
        else []
    )

    for round_index, round_input in enumerate(rounds):
        previous_mass = float(mass.sum())
        if base_mode:
            mass = exact_round(mass, policy[round_index], solver, round_input)
        else:
            outcomes = (
                solver.round_outcomes(
                    round_input,
                    allow_rollover=round_index < len(rounds) - 1,
                )
                if specification.rollover
                else solver.round_outcomes(round_input)
            )
            following_payment = (
                rounds[round_index + 1].payment
                if round_index + 1 < len(rounds)
                else final_payment
            )
            next_mass = np.zeros_like(mass)
            states = product(*(range(size) for size in contingency_shape))
            for contingency_state in states if contingency_shape else [()]:
                current = mass[contingency_state]
                if not np.any(current):
                    continue
                current_policy = policy[(round_index,) + contingency_state]
                for scenario, (scenario_prices, probability) in enumerate(
                    zip(prices, price_probabilities)
                ):
                    delivered = delivery_step(
                        current,
                        current_policy,
                        scenario,
                        scenario_prices,
                        solver,
                        0
                        if specification.payment_deduction
                        else int(round_input.payment.amount),
                        delivery_tables,
                    )
                    next_mass += probability * production_step(
                        delivered,
                        solver,
                        outcomes,
                        contingency_state,
                        contingency_shape,
                        specification,
                        int(following_payment.amount),
                    )
            mass = next_mass
        survival = float(mass.sum())
        if survival > previous_mass + 1e-7:
            raise AssertionError("policy occupancy increased probability mass")
        survival_curve.append(survival)

    incompleteness_curve = [1.0 - value for value in survival_curve]
    incompleteness = float(np.mean(incompleteness_curve))
    result = {
        "metric_version": METRIC_VERSION,
        "contingency_combination": specification.combination,
        "incompleteness": incompleteness,
        "completeness": 1.0 - incompleteness,
        "survival_curve": survival_curve,
        "incompleteness_curve": incompleteness_curve,
        "p_ever_broke": incompleteness_curve[-1],
        "n_rounds": len(rounds),
    }
    if p_sat is not None:
        result["P_sat"] = float(p_sat)
        result["survived_but_unsatisfied"] = (
            1.0 - float(p_sat) - incompleteness_curve[-1]
        )
    return result
