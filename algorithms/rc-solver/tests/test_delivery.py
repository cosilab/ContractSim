import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import Environment
from tests.fixtures import load_environment_09
from transitions.delivery import DeliveryKernel


class DeliveryKernelTests(unittest.TestCase):
    def test_supports_all_benchmark_minimum_delivery_rules(self):
        cases = (
            (1, {7: 1.0}),
            (3, {5: 1.0 / 3.0, 6: 1.0 / 3.0, 7: 1.0 / 3.0}),
            (5, {3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2}),
        )
        for environment_id, probabilities in cases:
            with self.subTest(environment_id=environment_id):
                environment = Environment.from_json(
                    PROJECT_ROOT
                    / "envs"
                    / str(environment_id)
                    / "environment.json"
                )
                kernel = DeliveryKernel(
                    environment.probability_dist.delivery_distribution,
                    environment.supplier.max_inventory_per_resource,
                    environment.supplier.max_order_per_resource_per_round,
                )

                expected = np.zeros(kernel.number_of_inventory_states)
                for inventory, probability in probabilities.items():
                    expected[inventory] = probability
                np.testing.assert_allclose(kernel.table[0, 7], expected)
                np.testing.assert_allclose(kernel.table.sum(axis=-1), 1.0)

    def test_builds_environment_09_delivery_table(self):
        environment = Environment(load_environment_09())
        kernel = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            environment.supplier.max_inventory_per_resource,
            environment.supplier.max_order_per_resource_per_round,
        )

        self.assertEqual(kernel.table.shape, (11, 13, 11))
        np.testing.assert_allclose(kernel.table.sum(axis=-1), 1.0)
        np.testing.assert_allclose(kernel.table[0, 6, 3:7], np.full(4, 0.25))
        self.assertAlmostEqual(kernel.table[8, 6, 10], 1.0)
        with self.assertRaises(ValueError):
            kernel.table[0, 0, 0] = 0.0

    def test_rejects_unsupported_delivery_distributions(self):
        environment = Environment(load_environment_09())
        unsupported = replace(
            environment.probability_dist.delivery_distribution,
            distribution="poisson",
        )

        with self.assertRaisesRegex(ValueError, "discrete_uniform"):
            DeliveryKernel(unsupported, 10, 12)

    def test_rejects_unsupported_minimum_delivery_rules(self):
        environment = Environment(load_environment_09())
        unsupported = replace(
            environment.probability_dist.delivery_distribution,
            minimum_received="ordered-1",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported delivery minimum rule"):
            DeliveryKernel(unsupported, 10, 12)

    def test_rejects_invalid_inventory_and_order_bounds(self):
        distribution = Environment(
            load_environment_09()
        ).probability_dist.delivery_distribution

        with self.assertRaisesRegex(ValueError, "max_inventory_per_resource"):
            DeliveryKernel(distribution, -1, 12)
        with self.assertRaisesRegex(ValueError, "max_order_per_resource_per_round"):
            DeliveryKernel(distribution, 10, 1.5)


if __name__ == "__main__":
    unittest.main()
