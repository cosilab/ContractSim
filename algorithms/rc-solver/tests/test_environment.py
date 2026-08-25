import copy
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import CustomerSpec, Environment, ProbabilityDist, SupplierSpec
from tests.fixtures import ENVIRONMENT_09_PATH, load_environment_09


class EnvironmentTests(unittest.TestCase):
    def test_builds_the_complete_environment_09_model(self):
        environment = Environment(load_environment_09())

        self.assertEqual(environment.schema_version, 1)
        self.assertEqual(environment.environment_id, 9)
        self.assertEqual(
            environment.environment_name,
            "env-09_baseline_current_cap20",
        )
        self.assertEqual(environment.random_seed, 42)
        self.assertIsInstance(environment.customer, CustomerSpec)
        self.assertIsInstance(environment.supplier, SupplierSpec)
        self.assertIsInstance(environment.probability_dist, ProbabilityDist)
        self.assertEqual(environment.customer.private_values["B"], 6)
        self.assertEqual(environment.supplier.starting_capital, 20)
        self.assertEqual(
            environment.probability_dist.delivery_distribution.distribution,
            "discrete_uniform",
        )
        self.assertEqual(
            environment.supported_contingencies,
            ("substitution", "payment_deduction", "rollover", "grim_trigger"),
        )

    def test_loads_environment_09_directly_from_json(self):
        environment = Environment.from_json(ENVIRONMENT_09_PATH)

        self.assertEqual(environment.environment_id, 9)
        self.assertEqual(environment.axes["operating_conditions"], "baseline")
        self.assertEqual(
            environment.probability_dist.kernel_id,
            "operating-baseline-v1",
        )

    def test_defensively_copies_metadata(self):
        source = load_environment_09()
        environment = Environment(source)

        source["axes"]["operating_conditions"] = "changed"

        self.assertEqual(environment.axes["operating_conditions"], "baseline")
        with self.assertRaises(TypeError):
            environment.axes["operating_conditions"] = "changed"

    def test_only_accepts_json_shaped_constructor_input(self):
        valid = Environment(load_environment_09())

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            Environment(valid)
        with self.assertRaises(TypeError):
            Environment(
                schema_version=valid.schema_version,
                environment_id=valid.environment_id,
                environment_name=valid.environment_name,
                customer=valid.customer,
                supplier=valid.supplier,
                probability_dist=valid.probability_dist,
                axes=valid.axes,
            )

    def test_rejects_cross_component_mismatches(self):
        source = copy.deepcopy(load_environment_09())
        for mapping in (
            source["base_game"]["customer"][
                "minimum_dishes_per_production_round"
            ],
            source["customer_preference"]["private_values"],
            source["customer_preference"]["substitution_soup_equivalents"],
        ):
            mapping["D"] = mapping.pop("C")

        with self.assertRaisesRegex(ValueError, "Customer products must match"):
            Environment(source)

        source = copy.deepcopy(load_environment_09())
        source["probability_kernel"]["kernel_id"] = "wrong-kernel"
        with self.assertRaisesRegex(ValueError, "does not match"):
            Environment(source)


if __name__ == "__main__":
    unittest.main()
