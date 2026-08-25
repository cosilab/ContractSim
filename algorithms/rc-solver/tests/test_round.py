import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import Environment
from tests.fixtures import load_environment_09
from transitions.delivery import DeliveryKernel
from transitions.resource import ResourceTransitionKernel
from transitions.round import RoundTransitionKernels
from transitions.spoilage import SpoilageKernel


def resource_kernel(need: int) -> ResourceTransitionKernel:
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
    return ResourceTransitionKernel(delivery, need, spoilage)


class RoundTransitionKernelsTests(unittest.TestCase):
    def test_validates_and_defensively_freezes_resource_selection(self):
        needs = {"X": 1}
        kernels = {"X": resource_kernel(1)}

        round_kernels = RoundTransitionKernels(2, needs, kernels)
        needs["X"] = 0
        kernels.clear()

        self.assertEqual(dict(round_kernels.resource_needs), {"X": 1})
        self.assertEqual(round_kernels.kernels_by_resource["X"].need, 1)
        with self.assertRaises(TypeError):
            round_kernels.resource_needs["X"] = 0

    def test_rejects_missing_resources_and_incorrect_kernel_needs(self):
        with self.assertRaisesRegex(ValueError, "resources must match"):
            RoundTransitionKernels(2, {"X": 1}, {})
        with self.assertRaisesRegex(ValueError, "does not match the round"):
            RoundTransitionKernels(2, {"X": 1}, {"X": resource_kernel(0)})


if __name__ == "__main__":
    unittest.main()
