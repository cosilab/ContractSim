import copy
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contract import Contract
from environment import Environment
from tests.fixtures import load_environment_09, load_rollover_contract
from transitions.contract import ContractTransitionKernels


class ContractTransitionKernelsTests(unittest.TestCase):
    def test_derives_round_needs_and_reuses_kernels_by_unique_need(self):
        environment = Environment(load_environment_09())
        contract = Contract(load_rollover_contract())

        kernels = ContractTransitionKernels(environment, contract)

        self.assertEqual(kernels.resource_aliases, ("X", "Y", "Z"))
        self.assertEqual(set(kernels.kernels_by_need), {2, 3, 4})
        self.assertEqual(len(kernels.rounds), 5)
        self.assertEqual(
            dict(kernels.for_week(2).resource_needs),
            {"X": 4, "Y": 4, "Z": 3},
        )
        self.assertEqual(
            dict(kernels.for_week(4).resource_needs),
            {"X": 2, "Y": 2, "Z": 4},
        )
        self.assertIs(
            kernels.for_week(2).kernels_by_resource["X"],
            kernels.for_week(2).kernels_by_resource["Y"],
        )
        self.assertIs(
            kernels.for_week(2).kernels_by_resource["X"],
            kernels.for_week(4).kernels_by_resource["Z"],
        )

    def test_uses_the_contracts_inferred_number_of_production_weeks(self):
        source = load_rollover_contract()
        source["production_schedule"] = source["production_schedule"][:2]
        source["payment_schedule"] = source["payment_schedule"][:3]
        contract = Contract(source)

        kernels = ContractTransitionKernels(
            Environment(load_environment_09()),
            contract,
        )

        self.assertEqual(contract.number_of_weeks, 5)
        self.assertEqual(tuple(round_.week for round_ in kernels.rounds), (2, 4))

    def test_is_immutable(self):
        kernels = ContractTransitionKernels(
            Environment(load_environment_09()),
            Contract(load_rollover_contract()),
        )

        with self.assertRaises(TypeError):
            kernels.kernels_by_need[99] = kernels.kernels_by_need[2]
        with self.assertRaises(TypeError):
            kernels.rounds[0].resource_needs["X"] = 99
        with self.assertRaises(FrozenInstanceError):
            kernels.rounds = ()

    def test_rejects_contract_products_that_do_not_match_the_environment(self):
        source = copy.deepcopy(load_rollover_contract())
        for production_week in source["production_schedule"]:
            production_week["product_quantities"]["D"] = (
                production_week["product_quantities"].pop("C")
            )
        source["product_prices"]["D"] = source["product_prices"].pop("C")

        with self.assertRaisesRegex(ValueError, "must match environment products"):
            ContractTransitionKernels(
                Environment(load_environment_09()),
                Contract(source),
            )

    def test_rejects_unknown_production_weeks(self):
        kernels = ContractTransitionKernels(
            Environment(load_environment_09()),
            Contract(load_rollover_contract()),
        )

        with self.assertRaisesRegex(KeyError, "week 12"):
            kernels.for_week(12)


if __name__ == "__main__":
    unittest.main()
