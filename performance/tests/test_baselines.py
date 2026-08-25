from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from performance.baselines.customer import CustomerRC, CustomerRCC, CustomerRE
from performance.baselines.supplier import SupplierRC, SupplierRCC, SupplierRE
from performance.engine.environment import (
    EnvironmentEngine,
    load_performance_environment,
)
from performance.tests.helpers import sample_config, sample_terms, write_contract


class BaselineProfileTests(unittest.TestCase):
    def test_customer_and_supplier_profiles_follow_the_environment(self) -> None:
        terms = sample_terms()
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            for environment_id in range(1, 7):
                with self.subTest(environment_id=environment_id):
                    environment = load_performance_environment(environment_id)
                    customer = CustomerRE(
                        sample_config(environment_id),
                        terms,
                        environment_id=environment_id,
                    )
                    supplier = SupplierRE(
                        contract_dir.name,
                        contract_dir,
                        sample_config(environment_id),
                        terms,
                        environment_id=environment_id,
                    )
                    self.assertEqual(
                        customer.private_values,
                        environment.customer_private_values,
                    )
                    self.assertEqual(
                        supplier.private_values,
                        environment.customer_private_values,
                    )
                    self.assertEqual(
                        customer.default_min_qty,
                        environment.customer_minimum_quantities,
                    )
                    self.assertEqual(
                        supplier.default_min_qty,
                        environment.customer_minimum_quantities,
                    )


class CustomerBaselineTests(unittest.TestCase):
    def test_rc_re_and_rcc_have_their_basic_payment_behaviors(self) -> None:
        config = sample_config()
        terms = sample_terms()
        rc = CustomerRC(config, terms, environment_id=1)
        re = CustomerRE(config, terms, environment_id=1)
        rcc = CustomerRCC(config, terms, environment_id=1)

        self.assertEqual(rc.decide_payment(week=1, remaining_budget=200)[0], 10)
        self.assertEqual(re.decide_payment(week=1, remaining_budget=200)[0], 0)
        self.assertEqual(rcc.decide_payment(week=1, remaining_budget=200)[0], 10)

        rcc.observe_production(week=2, produced={"A": 0, "B": 0, "C": 0})
        self.assertTrue(rcc.suspended)
        self.assertEqual(rcc.decide_payment(week=3, remaining_budget=190)[0], 0)


class SupplierBaselineTests(unittest.TestCase):
    def test_grim_rc_produces_nothing_if_exact_fulfillment_is_impossible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            game = EnvironmentEngine(1)
            game.inventory = {"X": 0, "Y": 0, "Z": 1}
            rc = SupplierRC(
                contract_dir.name,
                contract_dir,
                sample_config(),
                sample_terms(),
                environment_id=1,
            )

            production, record = rc.decide_production(game, 2)

            self.assertEqual(production, {"A": 0, "B": 0, "C": 0})
            self.assertTrue(record["customer_may_terminate"])

    def test_rc_re_and_rcc_have_their_basic_execution_behaviors(self) -> None:
        config = sample_config()
        terms = sample_terms()
        with tempfile.TemporaryDirectory() as temporary:
            contract_dir = write_contract(Path(temporary) / "contract")
            game = EnvironmentEngine(1)
            game.apply_payment(10)

            rc = SupplierRC(
                contract_dir.name,
                contract_dir,
                config,
                terms,
                environment_id=1,
            )
            purchase, _record = rc.decide_purchases(game, 2)
            self.assertEqual(purchase, {"X": 2, "Y": 2, "Z": 2})
            game.apply_purchase(purchase)
            production, _record = rc.decide_production(game, 2)
            self.assertEqual(production, {"A": 1, "B": 1, "C": 1})

            re = SupplierRE(
                contract_dir.name,
                contract_dir,
                config,
                terms,
                environment_id=1,
            )
            self.assertEqual(
                re.decide_purchases(game, 2)[0],
                {"X": 0, "Y": 0, "Z": 0},
            )
            self.assertEqual(
                re.decide_production(game, 2)[0],
                {"A": 0, "B": 0, "C": 0},
            )

            rcc = SupplierRCC(
                contract_dir.name,
                contract_dir,
                config,
                terms,
                environment_id=1,
            )
            rcc.observe_payment(week=1, payment=0)
            self.assertTrue(rcc.customer_defaulted)
            self.assertEqual(
                rcc.decide_purchases(game, 2)[0],
                {"X": 0, "Y": 0, "Z": 0},
            )


if __name__ == "__main__":
    unittest.main()
