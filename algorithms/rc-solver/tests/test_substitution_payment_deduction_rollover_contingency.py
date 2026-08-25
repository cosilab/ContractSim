import sys
import unittest
from pathlib import Path

import jax.numpy as jnp


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import WeekValueState
from contingencies import (
    SubstitutionPaymentDeductionRolloverContingency,
    SubstitutionPaymentDeductionRolloverOneStepSolver,
)
from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09
from tests.test_solver import contract_source, round_input, single_resource_environment
from transitions import ContractTransitionKernels


class SubstitutionPaymentDeductionRolloverContingencyTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.environment = Environment(load_environment_09())
        self.zero_deficit = {"A": 0, "B": 0, "C": 0}

    def test_above_minimum_demand_can_be_completed_through_substitution(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 1},
            minimum_quantities={"A": 1, "B": 1, "C": 1},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 2, "B": 1, "C": 1},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 4, "Y": 4, "Z": 2},
        )

        self.assertTrue(outcome.complete_fulfilment)
        self.assertEqual(dict(outcome.production), {"A": 1, "B": 3, "C": 1})
        self.assertEqual(dict(outcome.served), {"A": 2, "B": 1, "C": 1})

    def test_carried_deficit_cannot_be_substituted(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 1},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.evaluate_production(
            target_quantities={"A": 0, "B": 0, "C": 0},
            previous_deficit={"A": 1, "B": 0, "C": 0},
            production={"A": 0, "B": 2, "C": 0},
        )

        self.assertEqual(dict(outcome.exact_requirement), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(dict(outcome.served), self.zero_deficit)
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 0, "C": 0})

    def test_failed_substitution_uses_physical_weighted_shortfall(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.evaluate_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            previous_deficit=self.zero_deficit,
            production={"A": 0, "B": 0, "C": 4},
        )

        self.assertEqual(dict(outcome.served), self.zero_deficit)
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 1, "C": 0})
        self.assertEqual(outcome.substitution_points_used, 0)
        self.assertEqual(outcome.weighted_shortfall, 15)

    def test_failed_substitution_cannot_rescue_oversized_physical_deficit(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 1, "B": 10, "C": 0},
            max_deficit=2,
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 3, "B": 2, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 0, "Y": 0, "Z": 4},
        )

        self.assertEqual(dict(outcome.production), self.zero_deficit)
        self.assertEqual(dict(outcome.shortfall), {"A": 3, "B": 2, "C": 0})
        self.assertEqual(outcome.substitution_points_used, 0)
        self.assertTrue(outcome.terminates)

    def test_survival_beats_lower_weighted_but_oversized_shortfall(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 1, "B": 10, "C": 0},
            max_deficit=2,
            minimum_quantities={"A": 3, "B": 2, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 3, "B": 2, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 2, "Y": 2, "Z": 1},
        )

        self.assertEqual(dict(outcome.production), {"A": 1, "B": 1, "C": 0})
        self.assertEqual(dict(outcome.shortfall), {"A": 2, "B": 1, "C": 0})
        self.assertTrue(outcome.survives)

    def test_incoming_deficit_shortfall_uses_zero_output_grim(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            previous_deficit={"A": 1, "B": 0, "C": 0},
            available_inventory={"X": 0, "Y": 0, "Z": 4},
        )

        self.assertTrue(outcome.terminates)
        self.assertEqual(dict(outcome.production), self.zero_deficit)
        self.assertEqual(outcome.customer_value, 0)

    def test_final_round_shortfall_uses_zero_output_grim(self):
        contingency = SubstitutionPaymentDeductionRolloverContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 0, "Y": 0, "Z": 4},
            allow_rollover=False,
        )

        self.assertTrue(outcome.terminates)
        self.assertEqual(dict(outcome.production), self.zero_deficit)
        self.assertEqual(outcome.customer_value, 0)


class SubstitutionPaymentDeductionRolloverBellmanTests(unittest.TestCase):
    def test_first_residual_is_credited_and_reduces_following_payment(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([2], [0, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        current_round = round_input(contract, transitions, 0)
        solver = SubstitutionPaymentDeductionRolloverOneStepSolver(
            environment,
            {"A": 2},
            max_deficit=2,
        )
        ones = jnp.ones(solver.state_shape, dtype=jnp.float32)
        zeros = jnp.zeros(solver.state_shape, dtype=jnp.float32)
        continuation = WeekValueState(ones, zeros, zeros, zeros)

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            continuation,
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


if __name__ == "__main__":
    unittest.main()
