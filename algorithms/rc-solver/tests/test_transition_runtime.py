import copy
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contract import Contract
from environment import Environment
from runtime import TransitionRuntime
from tests.fixtures import load_environment_09, load_rollover_contract


class TransitionRuntimeTests(unittest.TestCase):
    def test_reuses_environment_and_resource_kernels_across_contracts(self):
        environment = Environment(load_environment_09())
        first_contract = Contract(load_rollover_contract())
        second_source = copy.deepcopy(load_rollover_contract())
        second_source["production_schedule"] = second_source[
            "production_schedule"
        ][:2]
        second_source["production_schedule"][0]["product_quantities"] = {
            "A": 2,
            "B": 3,
            "C": 1,
        }
        second_source["payment_schedule"] = second_source["payment_schedule"][:3]
        second_contract = Contract(second_source)
        runtime = TransitionRuntime(environment)

        first = runtime.for_contract(first_contract)
        second = runtime.for_contract(second_contract)

        self.assertIs(first.delivery, runtime.delivery)
        self.assertIs(second.delivery, runtime.delivery)
        self.assertIs(first.spoilage, runtime.spoilage)
        self.assertIs(second.spoilage, runtime.spoilage)
        self.assertEqual(set(runtime.kernels_by_need), {2, 3, 4, 5})
        self.assertEqual(set(second.kernels_by_need), {2, 3, 4, 5})
        for need in (2, 3, 4):
            self.assertIs(
                second.kernels_by_need[need],
                first.kernels_by_need[need],
            )
        self.assertIs(second.kernels_by_need[5], runtime.kernels_by_need[5])

    def test_builds_only_new_needs_and_exposes_a_read_only_cache(self):
        runtime = TransitionRuntime(Environment(load_environment_09()))

        need_two = runtime.resource_kernel(2)
        self.assertIs(runtime.resource_kernel(2), need_two)
        need_five = runtime.resource_kernel(5)

        self.assertEqual(set(runtime.kernels_by_need), {2, 5})
        self.assertIs(runtime.kernels_by_need[5], need_five)
        with self.assertRaises(TypeError):
            runtime.kernels_by_need[6] = need_five

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            TransitionRuntime({})

        runtime = TransitionRuntime(Environment(load_environment_09()))
        with self.assertRaises(TypeError):
            runtime.for_contract({})
        with self.assertRaises(ValueError):
            runtime.resource_kernel(-1)


if __name__ == "__main__":
    unittest.main()
