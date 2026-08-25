import copy
import sys
import unittest
from itertools import product
from pathlib import Path

import jax
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import OneStepSolver, ProductionRoundInput, WeekValueState
from bellman.solver import _best_order_index
from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09
from transitions import ContractTransitionKernels


def single_resource_environment(*, weighted_prices: bool = False) -> Environment:
    source = copy.deepcopy(load_environment_09())
    source["base_game"]["customer"][
        "minimum_dishes_per_production_round"
    ] = {"A": 1}
    source["base_game"]["recipes"] = {"A": {"X": 1}}
    source["base_game"]["inventory"]["max_units_per_ingredient"] = 1
    source["base_game"]["inventory"][
        "max_order_per_ingredient_per_round"
    ] = 2
    source["customer_preference"]["private_values"] = {"A": 12}
    source["customer_preference"]["substitution_soup_equivalents"] = {
        "A": 4
    }
    source["caterer_capital"]["starting_balance"] = 2
    source["caterer_capital"]["n_budget_states"] = 8
    source["operating_conditions"]["prices"]["ingredients"] = {
        "X": {
            "outcomes": (
                [
                    {"value": 1, "probability": 0.25},
                    {"value": 2, "probability": 0.75},
                ]
                if weighted_prices
                else [{"value": 1, "probability": 1.0}]
            )
        }
    }
    source["operating_conditions"]["spoilage"]["outcomes"] = [
        {"units_lost": 0, "probability": 1.0}
    ]
    return Environment(source)


def contract_source(
    quantities: list[int],
    payments: list[float],
) -> dict:
    return {
        "product_prices": {"A": 1},
        "production_schedule": [
            {
                "week": 2 * index + 2,
                "product_quantities": {"A": quantity},
            }
            for index, quantity in enumerate(quantities)
        ],
        "payment_schedule": [
            {"week": 2 * index + 1, "amount": amount}
            for index, amount in enumerate(payments)
        ],
    }


def three_resource_environment(*, weighted_prices: bool = False) -> Environment:
    source = copy.deepcopy(load_environment_09())
    source["base_game"]["inventory"]["max_units_per_ingredient"] = 1
    source["base_game"]["inventory"][
        "max_order_per_ingredient_per_round"
    ] = 1
    source["caterer_capital"]["starting_balance"] = 3
    source["caterer_capital"]["n_budget_states"] = 10
    price_specifications = source["operating_conditions"]["prices"][
        "ingredients"
    ]
    for specification in price_specifications.values():
        specification["outcomes"] = [{"value": 1, "probability": 1.0}]
    if weighted_prices:
        price_specifications["X"]["outcomes"] = [
            {"value": 1, "probability": 0.25},
            {"value": 2, "probability": 0.75},
        ]
    source["operating_conditions"]["spoilage"]["outcomes"] = [
        {"units_lost": 0, "probability": 1.0}
    ]
    return Environment(source)


def round_input(
    contract: Contract,
    transitions: ContractTransitionKernels,
    index: int,
) -> ProductionRoundInput:
    production = contract.production_schedule[index]
    return ProductionRoundInput(
        production=production,
        payment=contract.payment_schedule[index],
        transitions=transitions.for_week(production.week),
    )


def dense_reference_step(
    solver: OneStepSolver,
    current_round: ProductionRoundInput,
    continuation: WeekValueState,
    satisfaction_threshold: float,
) -> np.ndarray:
    """Small NumPy reference that evaluates all metrics for every order."""
    transition_tables = [
        current_round.transitions.kernels_by_resource[
            resource
        ].successful_table
        for resource in solver.resource_names
    ]
    payment = float(current_round.payment.amount)
    round_value = float(
        sum(
            current_round.production.product_quantities[product]
            * solver.environment.customer.private_values[product]
            for product in solver.environment.supplier.products
        )
    )
    continuation_values = np.asarray(continuation.stacked())
    output = np.zeros((4,) + solver.state_shape, dtype=np.float32)
    inventory_states = range(solver.number_of_inventory_states)
    order_states = range(solver.number_of_order_states)

    for current_inventory in product(
        inventory_states,
        repeat=len(solver.resource_names),
    ):
        for capital in range(solver.number_of_capital_states):
            selected_by_scenario: list[np.ndarray] = []
            for prices in solver.price_kernel.scenarios:
                candidates: list[np.ndarray] = []
                for orders in product(
                    order_states,
                    repeat=len(solver.resource_names),
                ):
                    cost = float(sum(price * order for price, order in zip(prices, orders)))
                    next_capital = int(capital + payment - cost)
                    if next_capital < 0:
                        candidates.append(
                            np.asarray((-1.0, -np.inf, 0.0, 0.0))
                        )
                        continue
                    next_capital = min(
                        next_capital,
                        solver.number_of_capital_states - 1,
                    )
                    expected = np.zeros(4, dtype=np.float64)
                    for next_inventory in product(
                        inventory_states,
                        repeat=len(solver.resource_names),
                    ):
                        probability = float(
                            np.prod(
                                [
                                    table[current, order, nxt]
                                    for table, current, order, nxt in zip(
                                        transition_tables,
                                        current_inventory,
                                        orders,
                                        next_inventory,
                                    )
                                ]
                            )
                        )
                        expected += probability * continuation_values[
                            (slice(None),) + next_inventory + (next_capital,)
                        ]
                    candidate = np.asarray(
                        (
                            expected[0],
                            payment - cost + expected[1],
                            expected[2] + expected[0] * round_value,
                            payment + expected[3],
                        )
                    )
                    candidates.append(candidate)
                candidate_array = np.asarray(candidates, dtype=np.float32)
                satisfactory = (
                    candidate_array[:, 0]
                    >= satisfaction_threshold - 1e-4
                )
                score = np.where(
                    satisfactory,
                    np.float32(1e6) + candidate_array[:, 1],
                    candidate_array[:, 0],
                )
                best = np.argmax(score)
                selected_by_scenario.append(candidates[int(best)])
            output[(slice(None),) + current_inventory + (capital,)] = np.average(
                selected_by_scenario,
                axis=0,
                weights=solver.price_kernel.probabilities,
            )
    return output


class OneStepSolverTests(unittest.TestCase):
    def test_contractbench_score_preserves_float32_near_tie(self):
        best = _best_order_index(
            jax.numpy.asarray([[[0.9], [0.9]]], dtype=jax.numpy.float32),
            jax.numpy.asarray([[[1.0], [1.03]]], dtype=jax.numpy.float32),
            jax.numpy.asarray(0.8, dtype=jax.numpy.float32),
            current_inventory_shape=(1,),
            number_of_resources=1,
            number_of_capital_states=1,
        )

        self.assertEqual(int(best[0, 0]), 0)

    def test_contractbench_score_preserves_extreme_utility_behavior(self):
        best = _best_order_index(
            jax.numpy.asarray([[[0.9], [0.89]]], dtype=jax.numpy.float32),
            jax.numpy.asarray([[[-2e6], [0.0]]], dtype=jax.numpy.float32),
            jax.numpy.asarray(0.9, dtype=jax.numpy.float32),
            current_inventory_shape=(1,),
            number_of_resources=1,
            number_of_capital_states=1,
        )

        self.assertEqual(int(best[0, 0]), 1)

    def test_terminal_state_adds_the_last_payment(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([1], [0, 5]))
        solver = OneStepSolver(environment)

        terminal = solver.terminal_state(contract.payment_schedule[-1])

        self.assertEqual(terminal.shape, (2, 8))
        self.assertTrue(
            all(isinstance(array, jax.Array) for array in terminal)
        )
        self.assertEqual(len(jax.tree.leaves(terminal)), 4)
        np.testing.assert_allclose(terminal.satisfaction_probability, 1.0)
        np.testing.assert_allclose(terminal.supplier_utility, 5.0)
        np.testing.assert_allclose(terminal.customer_value, 0.0)
        np.testing.assert_allclose(terminal.total_paid, 5.0)
        with self.assertRaises(TypeError):
            terminal.supplier_utility[0, 0] = 0.0

    def test_selects_orders_by_satisfaction_then_profit(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([1], [0, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        terminal = solver.terminal_state(contract.payment_schedule[-1])

        result = solver.solve(
            round_input(contract, transitions, 0),
            terminal,
            satisfaction_threshold=0.9,
        )

        # With capital 2, ordering 2 guarantees delivery and leaves $3 profit.
        self.assertAlmostEqual(result.satisfaction_probability[0, 2], 1.0)
        self.assertAlmostEqual(result.supplier_utility[0, 2], 3.0)
        self.assertAlmostEqual(result.customer_value[0, 2], 12.0)
        self.assertAlmostEqual(result.total_paid[0, 2], 5.0)

        # With capital 1, no action reaches 0.9, so the best-probability order
        # is chosen: order 1 succeeds with probability one half.
        self.assertAlmostEqual(result.satisfaction_probability[0, 1], 0.5)
        self.assertAlmostEqual(result.supplier_utility[0, 1], 1.5)
        self.assertAlmostEqual(result.customer_value[0, 1], 6.0)
        self.assertAlmostEqual(result.total_paid[0, 1], 2.5)

    def test_uses_declared_price_scenario_probabilities(self):
        environment = single_resource_environment(weighted_prices=True)
        contract = Contract(contract_source([1], [0, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        terminal = solver.terminal_state(contract.payment_schedule[-1])

        result = solver.solve(
            round_input(contract, transitions, 0),
            terminal,
            satisfaction_threshold=0.9,
        )

        # Order 2 costs $2 in the 0.25 scenario and $4 in the 0.75 scenario.
        self.assertAlmostEqual(result.satisfaction_probability[0, 4], 1.0)
        self.assertAlmostEqual(result.supplier_utility[0, 4], 1.5)

    def test_composes_all_environment_resource_transitions(self):
        environment = three_resource_environment()
        source = {
            "product_prices": {"A": 1, "B": 1, "C": 1},
            "production_schedule": [
                {
                    "week": 2,
                    "product_quantities": {"A": 1, "B": 0, "C": 0},
                }
            ],
            "payment_schedule": [
                {"week": 1, "amount": 0},
                {"week": 3, "amount": 5},
            ],
        }
        contract = Contract(source)
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        terminal = solver.terminal_state(contract.payment_schedule[-1])

        result = solver.solve(
            round_input(contract, transitions, 0),
            terminal,
            satisfaction_threshold=0.1,
        )

        initial_state = (0, 0, 0, 3)
        self.assertAlmostEqual(result.satisfaction_probability[initial_state], 0.125)
        self.assertAlmostEqual(result.supplier_utility[initial_state], -2.375)
        self.assertAlmostEqual(result.customer_value[initial_state], 1.5)
        self.assertAlmostEqual(result.total_paid[initial_state], 0.625)

    def test_round_matches_dense_contractbench_style_reference(self):
        environment = three_resource_environment(weighted_prices=True)
        source = {
            "product_prices": {"A": 1, "B": 1, "C": 1},
            "production_schedule": [
                {
                    "week": 2,
                    "product_quantities": {"A": 1, "B": 0, "C": 0},
                }
            ],
            "payment_schedule": [
                {"week": 1, "amount": 2},
                {"week": 3, "amount": 5},
            ],
        }
        contract = Contract(source)
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        state_grid = np.arange(
            np.prod(solver.state_shape),
            dtype=np.float32,
        ).reshape(solver.state_shape)
        continuation = WeekValueState(
            jax.numpy.asarray(0.2 + 0.1 * (state_grid % 7)),
            jax.numpy.asarray(-3.0 + 0.25 * state_grid),
            jax.numpy.asarray(1.5 * state_grid),
            jax.numpy.asarray(2.0 + 0.75 * state_grid),
        )
        current_round = round_input(contract, transitions, 0)

        result = solver.solve(
            current_round,
            continuation,
            satisfaction_threshold=0.55,
        )
        expected = dense_reference_step(
            solver,
            current_round,
            continuation,
            satisfaction_threshold=0.55,
        )

        np.testing.assert_allclose(
            np.asarray(result.stacked()),
            expected,
            atol=1e-5,
        )

    def test_steps_compose_over_an_inferred_contract_calendar(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([0, 0], [3, 4, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        terminal = solver.terminal_state(contract.payment_schedule[-1])
        state = terminal
        round_inputs = tuple(
            round_input(contract, transitions, index)
            for index in range(len(contract.production_schedule))
        )

        for index in reversed(range(len(contract.production_schedule))):
            state = solver.solve(
                round_inputs[index],
                state,
                satisfaction_threshold=0.9,
            )

        scanned = solver.solve_rounds(
            round_inputs,
            terminal,
            satisfaction_threshold=0.9,
        )

        self.assertEqual(contract.number_of_weeks, 5)
        self.assertAlmostEqual(state.satisfaction_probability[0, 2], 1.0)
        self.assertAlmostEqual(state.supplier_utility[0, 2], 12.0)
        self.assertAlmostEqual(state.customer_value[0, 2], 0.0)
        self.assertAlmostEqual(state.total_paid[0, 2], 12.0)
        np.testing.assert_allclose(scanned.stacked(), state.stacked(), atol=1e-6)

    def test_final_payment_helper_uses_generic_backward_scan(self):
        environment = single_resource_environment(weighted_prices=True)
        contract = Contract(contract_source([1, 1], [1, 2, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        solver = OneStepSolver(environment)
        round_inputs = tuple(
            round_input(contract, transitions, index)
            for index in range(len(contract.production_schedule))
        )

        generic = solver.solve_rounds(
            round_inputs,
            solver.terminal_state(contract.payment_schedule[-1]),
            satisfaction_threshold=0.9,
        )
        from_final_payment = solver.solve_rounds_from_final_payment(
            round_inputs,
            contract.payment_schedule[-1],
            satisfaction_threshold=0.9,
        )

        np.testing.assert_array_equal(
            from_final_payment.stacked(),
            generic.stacked(),
        )

if __name__ == "__main__":
    unittest.main()
