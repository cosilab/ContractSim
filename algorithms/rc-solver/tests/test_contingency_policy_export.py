import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import OneStepSolver, ProductionRoundInput
from contingencies import (
    PaymentDeductionOneStepSolver,
    PaymentDeductionRolloverOneStepSolver,
    RolloverOneStepSolver,
    SubstitutionOneStepSolver,
    SubstitutionPaymentDeductionOneStepSolver,
    SubstitutionPaymentDeductionRolloverOneStepSolver,
    SubstitutionRolloverOneStepSolver,
)
from contract import Contract
from tests.test_solver import contract_source, single_resource_environment
from transitions import ContractTransitionKernels


class ContingencyPolicyExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = single_resource_environment()
        self.contract = Contract(contract_source([1], [0, 5]))
        transitions = ContractTransitionKernels(self.environment, self.contract)
        production = self.contract.production_schedule[0]
        self.round_input = ProductionRoundInput(
            production=production,
            payment=self.contract.payment_schedule[0],
            transitions=transitions.for_week(production.week),
        )
        self.final_payment = self.contract.payment_schedule[-1]

    def assert_policy_matches_step(self, solver, expected, actual, policy):
        np.testing.assert_allclose(actual.stacked(), expected.stacked())
        self.assertEqual(
            policy.shape,
            solver.state_shape
            + (len(solver.price_kernel.scenarios), len(solver.resource_names)),
        )
        self.assertTrue(np.issubdtype(policy.dtype, np.integer))

    def test_substitution_policy(self):
        solver = SubstitutionOneStepSolver(self.environment)
        continuation = solver.terminal_state(self.final_payment)
        expected = solver.solve(self.round_input, continuation, 0.9)
        actual, policy = solver.solve_with_policy(
            self.round_input,
            continuation,
            0.9,
        )
        self.assert_policy_matches_step(solver, expected, actual, policy)

    def test_payment_deduction_policy(self):
        solver = PaymentDeductionOneStepSolver(self.environment, {"A": 2})
        continuation = solver.terminal_state()
        expected = solver.solve(
            self.round_input,
            self.final_payment,
            continuation,
            0.9,
        )
        actual, policy = solver.solve_with_policy(
            self.round_input,
            self.final_payment,
            continuation,
            0.9,
        )
        self.assert_policy_matches_step(solver, expected, actual, policy)

    def test_rollover_policy(self):
        solver = RolloverOneStepSolver(self.environment, max_deficit=2)
        continuation = solver.terminal_state(self.final_payment)
        expected = solver.solve(
            self.round_input,
            continuation,
            0.9,
            allow_rollover=False,
        )
        actual, policy = solver.solve_with_policy(
            self.round_input,
            continuation,
            0.9,
            allow_rollover=False,
        )
        self.assert_policy_matches_step(solver, expected, actual, policy)

    def test_payment_deduction_rollover_policy(self):
        solver = PaymentDeductionRolloverOneStepSolver(
            self.environment,
            {"A": 2},
            max_deficit=2,
        )
        continuation = solver.terminal_state()
        expected = solver.solve(
            self.round_input,
            self.final_payment,
            continuation,
            0.9,
            allow_rollover=False,
        )
        actual, policy = solver.solve_with_policy(
            self.round_input,
            self.final_payment,
            continuation,
            0.9,
            allow_rollover=False,
        )
        self.assert_policy_matches_step(solver, expected, actual, policy)

    def test_all_eight_suite_modes_complete_a_full_round(self):
        solvers = (
            OneStepSolver(self.environment),
            SubstitutionOneStepSolver(self.environment),
            PaymentDeductionOneStepSolver(self.environment, {"A": 2}),
            SubstitutionPaymentDeductionOneStepSolver(
                self.environment,
                {"A": 2},
            ),
            RolloverOneStepSolver(self.environment, max_deficit=2),
            SubstitutionRolloverOneStepSolver(
                self.environment,
                max_deficit=2,
            ),
            PaymentDeductionRolloverOneStepSolver(
                self.environment,
                {"A": 2},
                max_deficit=2,
            ),
            SubstitutionPaymentDeductionRolloverOneStepSolver(
                self.environment,
                {"A": 2},
                max_deficit=2,
            ),
        )

        for solver in solvers:
            with self.subTest(solver=type(solver).__name__):
                state, policy = (
                    solver.solve_rounds_from_final_payment_with_policy(
                        (self.round_input,),
                        self.final_payment,
                        0.9,
                    )
                )
                state.stacked().block_until_ready()
                self.assertEqual(policy.shape[0], 1)
                self.assertTrue(np.issubdtype(policy.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
