import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import Environment
from tests.fixtures import load_environment_09
from transitions.delivery import DeliveryKernel
from transitions.resource import ResourceTransitionKernel
from transitions.spoilage import SpoilageKernel


class ResourceTransitionKernelTests(unittest.TestCase):
    def test_composes_delivery_consumption_and_spoilage_with_failure(self):
        environment = Environment(load_environment_09())
        delivery = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            environment.supplier.max_inventory_per_resource,
            environment.supplier.max_order_per_resource_per_round,
        )
        spoilage = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            environment.supplier.max_inventory_per_resource,
        )
        kernel = ResourceTransitionKernel(delivery, 3, spoilage)

        self.assertEqual(kernel.table.shape, (11, 13, 12))
        self.assertEqual(kernel.successful_table.shape, (11, 13, 11))
        np.testing.assert_allclose(kernel.table.sum(axis=-1), 1.0)

        expected = np.zeros(12)
        expected[0] = 5.0 / 9.0
        expected[1] = 1.0 / 9.0
        expected[kernel.failure_state] = 1.0 / 3.0
        np.testing.assert_allclose(kernel.table[0, 4], expected)

    def test_rejects_component_state_spaces_that_do_not_match(self):
        environment = Environment(load_environment_09())
        delivery = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            2,
            2,
        )
        spoilage = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            3,
        )
        with self.assertRaisesRegex(ValueError, "spoilage inventory states"):
            ResourceTransitionKernel(delivery, 1, spoilage)

    def test_rejects_invalid_need(self):
        environment = Environment(load_environment_09())
        delivery = DeliveryKernel(
            environment.probability_dist.delivery_distribution,
            2,
            2,
        )
        spoilage = SpoilageKernel(
            environment.probability_dist.spoilage_distribution,
            2,
        )

        with self.assertRaisesRegex(ValueError, "resource need"):
            ResourceTransitionKernel(delivery, -1, spoilage)


if __name__ == "__main__":
    unittest.main()
