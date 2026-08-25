import sys
import unittest
from pathlib import Path

import jax.numpy as jnp


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import ProductionRoundInput, WeekValueState
from contingencies import (
    PaymentDeductionContingency,
    PaymentDeductionOneStepSolver,
)
from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09
from tests.test_solver import contract_source, round_input, single_resource_environment
from transitions import ContractTransitionKernels


class PaymentDeductionContingencyTests(unittest.TestCase):
    def test_weighted_shortfall_then_fewer_ingredients(self):
        source = load_environment_09()
        source["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ] = {"A": 0, "B": 0, "C": 0}
        environment = Environment(source)
        contingency = PaymentDeductionContingency(
            environment,
            {"A": 5, "B": 5, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            available_inventory={"X": 1, "Y": 1, "Z": 1},
        )

        self.assertEqual(dict(outcome.production), {"A": 0, "B": 1, "C": 0})
        self.assertEqual(outcome.weighted_shortfall, 5)
        self.assertEqual(sum(outcome.ingredient_usage.values()), 2)

    def test_full_fulfilment_wins_even_at_zero_deduction_price(self):
        source = load_environment_09()
        source["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ] = {"A": 0, "B": 0, "C": 1}
        environment = Environment(source)
        contingency = PaymentDeductionContingency(
            environment,
            {"A": 1, "B": 1, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 0, "B": 0, "C": 1},
            available_inventory={"X": 0, "Y": 0, "Z": 1},
        )

        self.assertTrue(outcome.complete_fulfilment)
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 1})

    def test_weekly_minimums_do_not_override_weighted_shortfall(self):
        environment = Environment(load_environment_09())
        contingency = PaymentDeductionContingency(
            environment,
            {"A": 0, "B": 1, "C": 10},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 2},
            available_inventory={"X": 2, "Y": 2, "Z": 2},
        )

        self.assertEqual(dict(outcome.production), {"A": 0, "B": 1, "C": 2})
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(outcome.weighted_shortfall, 0)

    def test_repeated_shortfall_uses_zero_output_grim(self):
        environment = single_resource_environment()
        contingency = PaymentDeductionContingency(environment, {"A": 2})

        grace = contingency.select_production(
            {"A": 2},
            {"X": 1},
        )
        repeated = contingency.select_production(
            {"A": 2},
            {"X": 1},
            previous_shortfall=True,
        )

        self.assertEqual(dict(grace.production), {"A": 1})
        self.assertEqual(grace.customer_value, 12)
        self.assertEqual(dict(repeated.production), {"A": 0})
        self.assertEqual(repeated.customer_value, 0)

class PaymentDeductionBellmanTests(unittest.TestCase):
    def _one_round(self, quantity: int, next_payment: int = 5):
        environment = single_resource_environment()
        contract = Contract(contract_source([quantity], [0, next_payment]))
        transitions = ContractTransitionKernels(environment, contract)
        current_round = round_input(contract, transitions, 0)
        solver = PaymentDeductionOneStepSolver(environment, {"A": 2})
        return contract, current_round, solver

    def test_first_shortfall_survives_but_repeated_shortfall_is_grim(self):
        contract, current_round, solver = self._one_round(2)

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            solver.terminal_state(),
            satisfaction_threshold=0.9,
        )

        grace_state = (0, 1, 0)
        self.assertEqual(result.satisfaction_probability[grace_state], 1)
        self.assertEqual(result.supplier_utility[grace_state], 3)
        self.assertEqual(result.customer_value[grace_state], 12)
        self.assertEqual(result.total_paid[grace_state], 3)

        repeated_state = (1, 1, 0)
        self.assertEqual(result.satisfaction_probability[repeated_state], 0)
        self.assertEqual(result.supplier_utility[repeated_state], 0)
        self.assertEqual(result.customer_value[repeated_state], 0)
        self.assertEqual(result.total_paid[repeated_state], 0)

    def test_weekly_minimum_failure_survives_only_as_a_first_shortfall(self):
        contract, current_round, solver = self._one_round(2)

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            solver.terminal_state(),
            satisfaction_threshold=0.9,
        )

        self.assertEqual(result.satisfaction_probability[0, 0, 0], 1)
        self.assertEqual(result.satisfaction_probability[1, 0, 0], 0)
        self.assertEqual(result.customer_value[0, 0, 0], 0)
        self.assertEqual(result.customer_value[1, 0, 0], 0)

    def test_deduction_is_added_with_next_payment_before_capital_lookup(self):
        contract, current_round, solver = self._one_round(2)
        capital = jnp.arange(solver.number_of_capital_states, dtype=jnp.float32)
        capital_values = jnp.broadcast_to(capital, solver.state_shape)
        continuation = WeekValueState(
            jnp.ones(solver.state_shape, dtype=jnp.float32),
            capital_values,
            jnp.zeros(solver.state_shape, dtype=jnp.float32),
            jnp.zeros(solver.state_shape, dtype=jnp.float32),
        )

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            continuation,
            satisfaction_threshold=0.9,
        )

        # B+ is zero, loss is 2, and the next payment is 5. The continuation
        # is therefore read at capital 3, and the effective payment also adds 3.
        self.assertEqual(result.supplier_utility[0, 1, 0], 6)

    def test_full_cure_resets_the_shortfall_flag(self):
        contract, current_round, solver = self._one_round(1, next_payment=0)
        probability = jnp.empty(solver.state_shape, dtype=jnp.float32)
        probability = probability.at[0].set(0.75)
        probability = probability.at[1].set(0.25)
        zeros = jnp.zeros(solver.state_shape, dtype=jnp.float32)
        continuation = WeekValueState(probability, zeros, zeros, zeros)

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            continuation,
            satisfaction_threshold=0.9,
        )

        self.assertEqual(result.satisfaction_probability[1, 1, 0], 0.75)

    def test_round_solver_credits_initial_payment_once(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([2], [2, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        inputs = (
            ProductionRoundInput(
                production=contract.production_schedule[0],
                payment=contract.payment_schedule[0],
                transitions=transitions.for_week(2),
            ),
        )
        solver = PaymentDeductionOneStepSolver(environment, {"A": 2})

        result = solver.solve_rounds_from_final_payment(
            inputs,
            contract.payment_schedule[-1],
            satisfaction_threshold=0.9,
        )

        self.assertEqual(result.supplier_utility[0, 0, 0], 3)
        self.assertEqual(result.total_paid[0, 0, 0], 3)

    def test_two_consecutive_shortfalls_terminate_on_the_second(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([2, 2], [0, 5, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        inputs = tuple(
            round_input(contract, transitions, index) for index in range(2)
        )
        solver = PaymentDeductionOneStepSolver(environment, {"A": 2})

        result = solver.solve_rounds_from_final_payment(
            inputs,
            contract.payment_schedule[-1],
            satisfaction_threshold=0.9,
        )

        initial_state = (0, 1, 0)
        self.assertEqual(result.satisfaction_probability[initial_state], 0)
        self.assertEqual(result.supplier_utility[initial_state], 3)
        self.assertEqual(result.customer_value[initial_state], 12)
        self.assertEqual(result.total_paid[initial_state], 3)


if __name__ == "__main__":
    unittest.main()
