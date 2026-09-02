from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
RC_SOLVER_ROOT = REPO_ROOT / "algorithms" / "rc-solver"
for directory in (SCRIPT_DIR, RC_SOLVER_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_contract_qual import (  # noqa: E402
    ContingencySpecification,
    deduplicate_jobs,
    make_contingency_solver,
    round_inputs,
    scalar_result,
)
from contract import Contract  # noqa: E402
from contract_completeness import occupancy_completeness  # noqa: E402
from environment import Environment  # noqa: E402
from runtime import TransitionRuntime  # noqa: E402


def small_environment() -> Environment:
    fixture = (
        RC_SOLVER_ROOT / "tests" / "data" / "environment_09.json"
    )
    source = copy.deepcopy(json.loads(fixture.read_text()))
    source["base_game"]["customer"][
        "minimum_dishes_per_production_round"
    ] = {"A": 1}
    source["base_game"]["recipes"] = {"A": {"X": 1}}
    source["base_game"]["inventory"]["max_units_per_ingredient"] = 1
    source["base_game"]["inventory"][
        "max_order_per_ingredient_per_round"
    ] = 2
    source["customer_preference"]["private_values"] = {"A": 12}
    source["customer_preference"]["substitution_soup_equivalents"] = {"A": 4}
    source["caterer_capital"]["starting_balance"] = 2
    source["caterer_capital"]["n_budget_states"] = 8
    source["operating_conditions"]["prices"]["ingredients"] = {
        "X": {"outcomes": [{"value": 1, "probability": 1.0}]}
    }
    source["operating_conditions"]["spoilage"]["outcomes"] = [
        {"units_lost": 0, "probability": 1.0}
    ]
    return Environment(source)


def small_contract() -> Contract:
    return Contract(
        {
            "product_prices": {"A": 2},
            "production_schedule": [
                {"week": 2, "product_quantities": {"A": 1}},
                {"week": 4, "product_quantities": {"A": 1}},
            ],
            "payment_schedule": [
                {"week": 1, "amount": 0},
                {"week": 3, "amount": 1},
                {"week": 5, "amount": 2},
            ],
        }
    )


def specification(
    substitution: bool,
    payment_deduction: bool,
    rollover: bool,
) -> ContingencySpecification:
    return ContingencySpecification(
        substitution=substitution,
        payment_deduction=payment_deduction,
        rollover=rollover,
        minimum_quantities={"A": 1},
        deduction_prices={"A": 2},
        max_deficit=2,
    )


class ContractQualTests(unittest.TestCase):
    def test_deduplication_escalates_a_shared_solve_to_policy_export(self) -> None:
        environment = small_environment()
        contract = small_contract()
        spec = specification(False, False, False)
        common = {
            "environment_id": environment.environment_id,
            "solve_hash": "same",
            "contract": {
                "product_prices": dict(contract.product_prices),
                "production_schedule": [
                    {
                        "week": row.week,
                        "product_quantities": dict(row.product_quantities),
                    }
                    for row in contract.production_schedule
                ],
                "payment_schedule": [
                    {"week": row.week, "amount": row.amount}
                    for row in contract.payment_schedule
                ],
            },
            "contingency": spec.as_mapping(),
            "satisfaction_threshold": 0.95,
        }
        jobs = deduplicate_jobs(
            [
                {**common, "record_id": "offer-1", "is_accepted_final": False},
                {**common, "record_id": "offer-2", "is_accepted_final": True},
            ],
            {environment.environment_id: environment},
        )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].record_ids, ["offer-1", "offer-2"])
        self.assertEqual(jobs[0].accepted_final_record_ids, ["offer-2"])
        self.assertTrue(jobs[0].needs_policy)

    def test_all_eight_dispatches_have_exact_policy_completeness(self) -> None:
        environment = small_environment()
        contract = small_contract()
        rounds = round_inputs(
            contract,
            TransitionRuntime(environment),
        )

        for flags in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (True, True, False),
            (False, False, True),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ):
            with self.subTest(flags=flags):
                spec = specification(*flags)
                solver = make_contingency_solver(environment, spec)
                backend = getattr(solver, "bellman_solver", solver)
                state, policies = (
                    backend.solve_rounds_from_final_payment_with_policy(
                        rounds,
                        contract.payment_schedule[-1],
                        0.95,
                    )
                )
                scalar = scalar_result(state, environment, spec)
                metric = occupancy_completeness(
                    policies,
                    solver=backend,
                    round_inputs=rounds,
                    final_payment=contract.payment_schedule[-1],
                    specification=spec,
                    starting_capital=environment.supplier.starting_capital,
                    p_sat=scalar["P_sat"],
                )

                self.assertEqual(metric["n_rounds"], 2)
                self.assertEqual(len(metric["survival_curve"]), 2)
                self.assertGreaterEqual(metric["completeness"], 0.0)
                self.assertLessEqual(metric["completeness"], 1.0)
                self.assertAlmostEqual(
                    metric["survival_curve"][-1],
                    scalar["P_sat"],
                    places=5,
                )


if __name__ == "__main__":
    unittest.main()
