from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from performance.baselines.supplier import SupplierRC, SupplierRCC
from performance.baselines.supplier._core import _delivery_profile
from performance.engine.environment import EnvironmentEngine
from performance.tests.helpers import sample_config, sample_terms, write_contract


class SupplierPolicyTests(unittest.TestCase):
    def test_policy_uses_balance_before_latest_payment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            policy = np.zeros((221, 3), dtype=np.int16)
            policy[20] = (1, 1, 1)
            policy[30] = (2, 2, 2)
            np.savez(
                contract_dir / "baselines" / "rcc" / "policy.npz",
                policies=policy,
                policy_axes=np.array(json.dumps(["budget", "ingredient"])),
                PRICE_RANGES=np.array([[1, 1, 2]], dtype=np.int64),
            )
            game = EnvironmentEngine(1)
            game.apply_payment(10)
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )
            supplier.observe_payment(week=1, payment=10)

            purchase, _record = supplier.decide_purchases(game, 2)

            self.assertEqual(purchase, {"X": 1, "Y": 1, "Z": 1})

    def test_deduction_policy_uses_balance_after_latest_payment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            policy = np.zeros((221, 3), dtype=np.int16)
            policy[20] = (1, 1, 1)
            policy[30] = (2, 2, 2)
            np.savez(
                contract_dir / "baselines" / "rcc" / "policy.npz",
                policies=policy,
                policy_axes=np.array(json.dumps(["budget", "ingredient"])),
                PRICE_RANGES=np.array([[1, 1, 2]], dtype=np.int64),
            )
            config = sample_config()
            config["contingencies"] = ["payment_deduction", "grim_trigger"]
            config["contingency_mode"] = "payment_deduction"
            game = EnvironmentEngine(1)
            game.apply_payment(10)
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                config,
                sample_terms(),
                environment_id=1,
            )
            supplier.observe_payment(week=1, payment=10)

            purchase, _record = supplier.decide_purchases(game, 2)

            self.assertEqual(purchase, {"X": 2, "Y": 2, "Z": 2})


class SupplierContingencyTests(unittest.TestCase):
    def test_rollover_selection_matches_ordinary_and_grim_states(self) -> None:
        config = sample_config()
        config["contingencies"] = ["rollover", "grim_trigger"]
        config["contingency_mode"] = "rollover"
        terms = sample_terms()
        for row in terms["production_schedule"]:
            row.update({"A": 2, "B": 2, "C": 1})

        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")

            ordinary = SupplierRC(
                contract_dir.name, contract_dir, config, terms, environment_id=1
            )
            game = EnvironmentEngine(1)
            game.inventory = {"X": 1, "Y": 1, "Z": 1}
            self.assertEqual(
                ordinary.decide_production(game, 2)[0],
                {"A": 0, "B": 1, "C": 1},
            )

            carried = SupplierRC(
                contract_dir.name, contract_dir, config, terms, environment_id=1
            )
            carried.carry_deficit = {"A": 1, "B": 0, "C": 0}
            game.inventory = {"X": 0, "Y": 0, "Z": 1}
            self.assertEqual(
                carried.decide_production(game, 4)[0],
                {"A": 0, "B": 0, "C": 0},
            )

            final = SupplierRC(
                contract_dir.name, contract_dir, config, terms, environment_id=1
            )
            self.assertEqual(
                final.decide_production(game, 10)[0],
                {"A": 0, "B": 0, "C": 0},
            )

    def test_repeated_payment_deduction_shortfall_produces_nothing(self) -> None:
        config = sample_config()
        config["contingencies"] = ["payment_deduction", "grim_trigger"]
        config["contingency_mode"] = "payment_deduction"

        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            supplier = SupplierRC(
                contract_dir.name,
                contract_dir,
                config,
                sample_terms(),
                environment_id=1,
            )
            game = EnvironmentEngine(1)
            game.inventory = {"X": 0, "Y": 0, "Z": 1}

            first, _record = supplier.decide_production(game, 2)
            second, record = supplier.decide_production(game, 4)

            self.assertEqual(first, {"A": 0, "B": 0, "C": 1})
            self.assertEqual(second, {"A": 0, "B": 0, "C": 0})
            self.assertTrue(record["customer_may_terminate"])

    def test_substitution_grim_requires_complete_fulfillment(self) -> None:
        config = sample_config()
        config["contingencies"] = ["substitution", "grim_trigger"]
        terms = sample_terms()
        terms["production_schedule"][0].update({"A": 2, "B": 2, "C": 2})

        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            cases = (
                ({"X": 6, "Y": 6, "Z": 2}, {"A": 1, "B": 5, "C": 1}),
                ({"X": 0, "Y": 0, "Z": 1}, {"A": 0, "B": 0, "C": 0}),
            )
            for inventory, expected in cases:
                with self.subTest(inventory=inventory):
                    game = EnvironmentEngine(1)
                    game.inventory = inventory
                    supplier = SupplierRC(
                        contract_dir.name,
                        contract_dir,
                        config,
                        terms,
                        environment_id=1,
                    )
                    production, _record = supplier.decide_production(game, 2)
                    self.assertEqual(production, expected)

    def test_incomplete_combined_substitution_uses_physical_shortfall(self) -> None:
        profile = _delivery_profile(
            produced={"A": 1, "B": 4, "C": 1},
            target={"A": 2, "B": 2, "C": 2},
            exact_req={"A": 1, "B": 1, "C": 1},
            allow_substitution=True,
            require_complete_substitution=True,
            weights={"A": 12, "B": 6, "C": 3},
        )

        self.assertEqual(profile["shortfall"], {"A": 1, "B": 0, "C": 1})
        self.assertEqual(profile["shortfall_value"], 15)

    def test_customer_exercises_termination_after_supplier_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )
            profile = _delivery_profile(
                produced={"A": 0, "B": 0, "C": 0},
                target={"A": 1, "B": 1, "C": 1},
                exact_req={"A": 1, "B": 1, "C": 1},
                allow_substitution=False,
                weights={"A": 12, "B": 6, "C": 3},
            )
            supplier._advance_contingency_state(profile)

            observation = supplier.observe_payment(week=3, payment=0)

            self.assertTrue(supplier.customer_may_terminate)
            self.assertTrue(supplier.contract_terminated)
            self.assertTrue(supplier.customer_payments_suspended)
            self.assertFalse(supplier.customer_defaulted)
            self.assertEqual(observation["legally_due_payment"], 0.0)
            self.assertEqual(
                observation["reason"],
                "customer_exercised_termination_after_supplier_default",
            )
            _purchase, record = supplier.decide_purchases(
                EnvironmentEngine(1),
                4,
            )
            self.assertEqual(record["reason"], "contract_terminated")

    def test_full_payment_does_not_exercise_available_termination_right(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )
            supplier.customer_may_terminate = True

            supplier.observe_payment(week=3, payment=10)

            self.assertFalse(supplier.contract_terminated)
            self.assertFalse(supplier.customer_defaulted)

    def test_underpayment_without_supplier_violation_is_customer_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )

            observation = supplier.observe_payment(week=1, payment=9)

            self.assertTrue(supplier.customer_defaulted)
            self.assertFalse(supplier.contract_terminated)
            self.assertEqual(observation["customer_payment_shortfall"], 1.0)

    def test_zero_payment_is_compliant_when_deduction_covers_due(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            supplier = SupplierRCC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )
            supplier.customer_may_terminate = True
            supplier.pending_deduction = 10

            observation = supplier.observe_payment(week=3, payment=0)

            self.assertFalse(supplier.contract_terminated)
            self.assertFalse(supplier.customer_defaulted)
            self.assertEqual(observation["legally_due_payment"], 0.0)


if __name__ == "__main__":
    unittest.main()
