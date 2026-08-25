import copy
import sys
import unittest
from pathlib import Path

import jax.numpy as jnp


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman import ProductionRoundInput, WeekValueState
from contingencies import (
    SubstitutionPaymentDeductionContingency,
    SubstitutionPaymentDeductionOneStepSolver,
)
from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09
from transitions import ContractTransitionKernels


def two_product_environment() -> Environment:
    source = copy.deepcopy(load_environment_09())
    source["base_game"]["customer"][
        "minimum_dishes_per_production_round"
    ] = {"A": 0, "B": 0}
    source["base_game"]["recipes"] = {
        "A": {"X": 2},
        "B": {"X": 1},
    }
    source["base_game"]["inventory"]["max_units_per_ingredient"] = 3
    source["base_game"]["inventory"][
        "max_order_per_ingredient_per_round"
    ] = 3
    source["customer_preference"]["private_values"] = {"A": 12, "B": 7}
    source["customer_preference"]["substitution_soup_equivalents"] = {
        "A": 2,
        "B": 1,
    }
    source["caterer_capital"]["starting_balance"] = 3
    source["caterer_capital"]["n_budget_states"] = 10
    source["operating_conditions"]["prices"]["ingredients"] = {
        "X": {"outcomes": [{"value": 1, "probability": 1.0}]}
    }
    source["operating_conditions"]["spoilage"]["outcomes"] = [
        {"units_lost": 0, "probability": 1.0}
    ]
    return Environment(source)


def one_round_input(
    environment: Environment,
    quantities: dict[str, int],
    final_payment: int,
) -> tuple[Contract, ProductionRoundInput]:
    contract = Contract(
        {
            "product_prices": {"A": 2, "B": 1},
            "production_schedule": [
                {"week": 2, "product_quantities": quantities}
            ],
            "payment_schedule": [
                {"week": 1, "amount": 0},
                {"week": 3, "amount": final_payment},
            ],
        }
    )
    transitions = ContractTransitionKernels(environment, contract)
    production = contract.production_schedule[0]
    return contract, ProductionRoundInput(
        production=production,
        payment=contract.payment_schedule[0],
        transitions=transitions.for_week(production.week),
    )


class SubstitutionPaymentDeductionContingencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = Environment(load_environment_09())

    def test_failed_substitution_uses_physical_weighted_shortfall(self):
        contingency = SubstitutionPaymentDeductionContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.evaluate_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            production={"A": 0, "B": 0, "C": 4},
        )

        self.assertEqual(dict(outcome.served), {"A": 0, "B": 0, "C": 0})
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 1, "C": 0})
        self.assertEqual(outcome.substitution_points_created, 4)
        self.assertEqual(outcome.substitution_points_used, 0)
        self.assertEqual(outcome.weighted_shortfall, 15)

    def test_payded_fallback_selects_physical_plan_by_weighted_shortfall(self):
        contingency = SubstitutionPaymentDeductionContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            available_inventory={"X": 1, "Y": 1, "Z": 2},
        )

        self.assertEqual(dict(outcome.production), {"A": 0, "B": 1, "C": 0})
        self.assertEqual(dict(outcome.shortfall), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(outcome.weighted_shortfall, 5)
        self.assertEqual(outcome.substitution_points_used, 0)
        self.assertEqual(sum(outcome.ingredient_usage.values()), 2)

    def test_repeated_shortfall_uses_zero_output_grim(self):
        contingency = SubstitutionPaymentDeductionContingency(
            self.environment,
            {"A": 5, "B": 10, "C": 0},
            minimum_quantities={"A": 0, "B": 0, "C": 0},
        )

        outcome = contingency.select_production(
            target_quantities={"A": 1, "B": 1, "C": 0},
            available_inventory={"X": 0, "Y": 0, "Z": 4},
            previous_shortfall=True,
        )

        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 0})
        self.assertEqual(sum(outcome.ingredient_usage.values()), 0)
        self.assertEqual(outcome.customer_value, 0)


class SubstitutionPaymentDeductionBellmanTests(unittest.TestCase):
    def test_first_partial_delivery_is_credited_but_repeated_one_is_not(self):
        environment = two_product_environment()
        contract, current_round = one_round_input(
            environment,
            {"A": 1, "B": 1},
            final_payment=5,
        )
        solver = SubstitutionPaymentDeductionOneStepSolver(
            environment,
            {"A": 2, "B": 1},
        )

        result = solver.solve(
            current_round,
            contract.payment_schedule[-1],
            solver.terminal_state(),
            satisfaction_threshold=0.9,
        )

        grace_state = (0, 1, 0)
        self.assertEqual(result.satisfaction_probability[grace_state], 1)
        self.assertEqual(result.supplier_utility[grace_state], 3)
        self.assertEqual(result.customer_value[grace_state], 7)
        self.assertEqual(result.total_paid[grace_state], 3)

        repeated_state = (1, 1, 0)
        self.assertEqual(result.satisfaction_probability[repeated_state], 0)
        self.assertEqual(result.supplier_utility[repeated_state], 0)
        self.assertEqual(result.customer_value[repeated_state], 0)
        self.assertEqual(result.total_paid[repeated_state], 0)

    def test_full_service_through_substitution_clears_hardship(self):
        environment = two_product_environment()
        contract, current_round = one_round_input(
            environment,
            {"A": 1, "B": 0},
            final_payment=0,
        )
        solver = SubstitutionPaymentDeductionOneStepSolver(
            environment,
            {"A": 2, "B": 1},
        )
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

        self.assertEqual(result.satisfaction_probability[1, 2, 0], 0.75)


if __name__ == "__main__":
    unittest.main()
