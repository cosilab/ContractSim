from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contract import Contract
from environment import Environment
from rc_solver import RCCSolver
from tests.fixtures import load_environment_09


def single_resource_environment() -> dict:
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
        "X": {"outcomes": [{"value": 1, "probability": 1.0}]}
    }
    source["operating_conditions"]["spoilage"]["outcomes"] = [
        {"units_lost": 0, "probability": 1.0}
    ]
    return source


def contract_source(quantities: list[int], payments: list[float]) -> dict:
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


def run_solver(
    environment_source: dict,
    contract: dict,
    satisfaction_threshold: float | None = None,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        environment_path = directory / "environment.json"
        contract_path = directory / "contract.json"
        environment_path.write_text(json.dumps(environment_source))
        contract_path.write_text(json.dumps(contract))
        command = [
            sys.executable,
            str(CODE_DIR / "rc_solver.py"),
            "--environment",
            str(environment_path),
            "--contract",
            str(contract_path),
        ]
        if satisfaction_threshold is not None:
            command.extend(("--sat-thresh", str(satisfaction_threshold)))
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


class ModularSolverTests(unittest.TestCase):
    def test_reusable_api_reports_the_same_three_values(self):
        solver = RCCSolver(Environment(single_resource_environment()))
        result = solver.solve(
            Contract(contract_source([1], [0, 5])),
            satisfaction_threshold=0.9,
        )

        self.assertEqual(
            set(result),
            {"P_sat", "E_supplier_utility", "E_customer_utility"},
        )
        self.assertAlmostEqual(result["P_sat"], 1.0)
        self.assertAlmostEqual(result["E_supplier_utility"], 3.0)
        self.assertAlmostEqual(result["E_customer_utility"], 207.0)

    def test_reusable_api_retains_runtime_objects_and_need_cache(self):
        solver = RCCSolver(Environment(single_resource_environment()))
        bellman_solver = solver.bellman_solver
        transition_runtime = solver.transition_runtime

        solver.solve(Contract(contract_source([1], [0, 5])))
        solver.solve(Contract(contract_source([2], [0, 5])))

        self.assertIs(solver.bellman_solver, bellman_solver)
        self.assertIs(solver.transition_runtime, transition_runtime)
        self.assertEqual(set(transition_runtime.kernels_by_need), {1, 2})

    def test_saved_policy_forward_trace_matches_customer_utility(self):
        environment_source = single_resource_environment()
        environment_source["base_game"]["inventory"][
            "max_order_per_ingredient_per_round"
        ] = 1
        solver = RCCSolver(Environment(environment_source))
        contract = Contract(contract_source([1, 1], [0, 0, 0]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            policy_path = Path(temporary_directory) / "policy.npz"
            result = solver.solve(
                contract,
                satisfaction_threshold=0.9,
                policy_path=policy_path,
            )
            with np.load(policy_path, allow_pickle=False) as saved:
                policies = np.asarray(saved["policies"])

        transitions = solver.transition_runtime.for_contract(contract)
        probability_mass = np.zeros((2, 8), dtype=np.float64)
        probability_mass[0, 2] = 1.0
        survival_probabilities = []
        for round_index, (payment, production) in enumerate(zip(
            contract.payment_schedule,
            contract.production_schedule,
        )):
            success_table = transitions.for_week(
                production.week
            ).kernels_by_resource["X"].successful_table
            next_mass = np.zeros_like(probability_mass)
            for inventory, capital in np.ndindex(probability_mass.shape):
                order = int(policies[round_index, inventory, capital, 0, 0])
                next_capital = min(
                    capital + int(payment.amount) - order,
                    probability_mass.shape[1] - 1,
                )
                if next_capital < 0:
                    continue
                next_mass[:, next_capital] += (
                    probability_mass[inventory, capital]
                    * success_table[inventory, order]
                )
            probability_mass = next_mass
            survival_probabilities.append(float(probability_mass.sum()))

        round_values = [
            sum(
                quantity * solver.environment.customer.private_values[product]
                for product, quantity in production.product_quantities.items()
            )
            for production in contract.production_schedule
        ]
        traced_customer_value = sum(
            probability * value
            for probability, value in zip(survival_probabilities, round_values)
        )

        # Each round succeeds with probability 0.5, so completed-round value is
        # 12 * 0.5 + 12 * 0.25 = 9 rather than 24 * 0.25 = 6.
        np.testing.assert_allclose(survival_probabilities, [0.5, 0.25])
        self.assertAlmostEqual(traced_customer_value, 9.0)
        self.assertAlmostEqual(result["P_sat"], 0.25)
        self.assertAlmostEqual(result["E_supplier_utility"], -1.5)
        self.assertAlmostEqual(
            result["E_customer_utility"],
            solver.environment.customer.starting_budget + traced_customer_value,
        )

    def test_cli_solves_one_round_and_reports_three_values(self):
        environment = single_resource_environment()
        result = run_solver(
            environment,
            contract_source([1], [0, 5]),
            satisfaction_threshold=0.9,
        )

        self.assertEqual(
            set(result),
            {"P_sat", "E_supplier_utility", "E_customer_utility"},
        )
        self.assertAlmostEqual(result["P_sat"], 1.0)
        self.assertAlmostEqual(result["E_supplier_utility"], 3.0)
        self.assertAlmostEqual(result["E_customer_utility"], 207.0)

    def test_cli_optionally_saves_decoded_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            environment_path = directory / "environment.json"
            contract_path = directory / "contract.json"
            policy_path = directory / "policy.npz"
            environment_path.write_text(json.dumps(single_resource_environment()))
            contract_path.write_text(json.dumps(contract_source([1], [0, 5])))

            subprocess.run(
                [
                    sys.executable,
                    str(CODE_DIR / "rc_solver.py"),
                    "--environment",
                    str(environment_path),
                    "--contract",
                    str(contract_path),
                    "--sat-thresh",
                    "0.9",
                    "--save-policy",
                    str(policy_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with np.load(policy_path, allow_pickle=False) as saved:
                policies = np.asarray(saved["policies"])
                axes = json.loads(str(saved["policy_axes"].item()))
                price_ranges = np.asarray(saved["PRICE_RANGES"])

            self.assertEqual(
                axes,
                ["round", "inv_x", "budget", "scenario", "ingredient"],
            )
            self.assertEqual(policies.shape, (1, 2, 8, 1, 1))
            self.assertEqual(policies.dtype, np.int16)
            self.assertEqual(int(policies[0, 0, 2, 0, 0]), 2)
            np.testing.assert_array_equal(price_ranges, [[1.0]])

    def test_runs_the_reverse_loop_for_a_variable_length_contract(self):
        environment = single_resource_environment()
        result = run_solver(
            environment,
            contract_source([0, 0], [3, 4, 5]),
            satisfaction_threshold=0.9,
        )

        self.assertAlmostEqual(result["P_sat"], 1.0)
        self.assertAlmostEqual(result["E_supplier_utility"], 12.0)
        self.assertAlmostEqual(result["E_customer_utility"], 188.0)

    def test_default_satisfaction_threshold_is_point_95(self):
        environment = single_resource_environment()
        contract = contract_source([1], [0, 5])

        default_result = run_solver(environment, contract)
        explicit_result = run_solver(
            environment,
            contract,
            satisfaction_threshold=0.95,
        )

        self.assertEqual(default_result, explicit_result)

if __name__ == "__main__":
    unittest.main()
