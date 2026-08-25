import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import Environment
from tests.fixtures import load_environment_09
from transitions.spoilage import SpoilageKernel


class SpoilageKernelTests(unittest.TestCase):
    def test_builds_environment_09_spoilage_table(self):
        environment = Environment(load_environment_09())
        kernel = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            environment.supplier.max_inventory_per_resource,
        )

        self.assertEqual(kernel.table.shape, (11, 11))
        np.testing.assert_allclose(kernel.table.sum(axis=-1), 1.0)
        np.testing.assert_allclose(kernel.table[2, :3], np.full(3, 1.0 / 3.0))
        with self.assertRaises(ValueError):
            kernel.table[0, 0] = 0.0

    def test_rejects_correlated_resource_spoilage(self):
        environment = Environment(load_environment_09())
        unsupported = replace(
            environment.probability_dist.spoilage_distribution,
            independent_across_resources=False,
        )

        with self.assertRaisesRegex(ValueError, "independent resource spoilage"):
            SpoilageKernel(unsupported, 10)

    def test_rejects_invalid_inventory_bounds(self):
        distribution = Environment(
            load_environment_09()
        ).probability_dist.spoilage_distribution

        with self.assertRaisesRegex(ValueError, "max_inventory_per_resource"):
            SpoilageKernel(distribution, -1)


if __name__ == "__main__":
    unittest.main()
