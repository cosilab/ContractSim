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
from transitions.price import PriceKernel


class PriceKernelTests(unittest.TestCase):
    def test_compiles_environment_09_joint_price_scenarios(self):
        environment = Environment(load_environment_09())
        kernel = PriceKernel(environment.probability_dist.price_distribution)

        self.assertEqual(kernel.resource_names, ("X", "Y", "Z"))
        self.assertEqual(kernel.scenarios.shape, (8, 3))
        self.assertEqual(kernel.probabilities.shape, (8,))
        np.testing.assert_allclose(kernel.scenarios[0], (1, 1, 2))
        np.testing.assert_allclose(kernel.scenarios[-1], (3, 2, 3))
        np.testing.assert_allclose(kernel.probabilities, np.full(8, 1.0 / 8.0))
        self.assertEqual(
            set(kernel.as_solver_mapping()),
            {"PRICE_RANGES", "PRICE_PROB"},
        )
        with self.assertRaises(ValueError):
            kernel.scenarios[0, 0] = 99

    def test_rejects_correlated_resource_prices(self):
        environment = Environment(load_environment_09())
        unsupported = replace(
            environment.probability_dist.price_distribution,
            independent_across_resources=False,
        )

        with self.assertRaisesRegex(ValueError, "independent resource prices"):
            PriceKernel(unsupported)


if __name__ == "__main__":
    unittest.main()
