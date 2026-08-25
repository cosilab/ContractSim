import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import ProductionRoundInput
from contingencies import SubstitutionContingency, SubstitutionOneStepSolver
from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09
from tests.test_solver import contract_source, single_resource_environment
from transitions import ContractTransitionKernels


class SubstitutionContingencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = Environment(load_environment_09())

    def test_minimum_dishes_cannot_be_substituted(self):
        contingency = SubstitutionContingency(self.environment)

        outcome = contingency.evaluate_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            production={"A": 0, "B": 3, "C": 0},
        )

        self.assertEqual(dict(outcome.served), {"A": 0, "B": 1, "C": 0})
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 0, "C": 0})
        self.assertFalse(outcome.satisfied)

    def test_point_allocation_uses_available_credit(self):
        contingency = SubstitutionContingency(
            self.environment,
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.evaluate_production(
            target_quantities={"A": 0, "B": 1, "C": 3},
            production={"A": 0, "B": 1, "C": 1},
        )

        self.assertEqual(outcome.substitution_points_created, 3)
        self.assertEqual(outcome.substitution_points_used, 3)
        self.assertEqual(dict(outcome.served), {"A": 0, "B": 0, "C": 3})

    def test_equal_service_prefers_fewer_ingredients(self):
        contingency = SubstitutionContingency(
            self.environment,
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 0, "C": 0},
            available_inventory={"X": 4, "Y": 4, "Z": 4},
        )

        self.assertEqual(dict(outcome.production), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(dict(outcome.ingredient_usage), {"X": 1, "Y": 1, "Z": 1})
        self.assertTrue(outcome.satisfied)

    def test_grim_failure_does_not_credit_partial_customer_value(self):
        environment = single_resource_environment()
        contract = Contract(contract_source([2], [0, 5]))
        transitions = ContractTransitionKernels(environment, contract)
        production = contract.production_schedule[0]
        round_input = ProductionRoundInput(
            production=production,
            payment=contract.payment_schedule[0],
            transitions=transitions.for_week(production.week),
        )
        solver = SubstitutionOneStepSolver(environment)

        outcomes = solver.round_outcomes(round_input)
        self.assertFalse(outcomes.complete_fulfilment[1])
        self.assertEqual(outcomes.customer_value[1], 12)

        result = solver.solve(
            round_input,
            solver.terminal_state(contract.payment_schedule[-1]),
            satisfaction_threshold=0.9,
        )

        self.assertEqual(result.customer_value[1, 0], 0)


if __name__ == "__main__":
    unittest.main()
