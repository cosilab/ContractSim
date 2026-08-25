import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment.customer import CustomerSpec
from tests.fixtures import load_environment_09


class CustomerSpecTests(unittest.TestCase):
    def test_loads_the_environment_shape_used_by_the_current_solver(self):
        spec = CustomerSpec(load_environment_09())

        self.assertEqual(spec.starting_budget, 200.0)
        self.assertEqual(dict(spec.private_values), {"A": 12, "B": 6, "C": 3})
        self.assertEqual(
            dict(spec.minimum_products_per_production_round),
            {"A": 1, "B": 1, "C": 1},
        )
        self.assertEqual(
            dict(spec.substitution_rules),
            {"A": 4, "B": 2, "C": 1},
        )

    def test_is_immutable_and_defensively_copies_input_mappings(self):
        environment = load_environment_09()
        spec = CustomerSpec(environment)

        environment["customer_preference"]["private_values"]["A"] = 999

        self.assertEqual(spec.private_values["A"], 12)
        with self.assertRaises(TypeError):
            spec.private_values["A"] = 999
        with self.assertRaises(FrozenInstanceError):
            spec.starting_budget = 0

    def test_rejects_customer_mappings_with_different_product_sets(self):
        environment = load_environment_09()
        del environment["customer_preference"]["private_values"]["C"]

        with self.assertRaisesRegex(ValueError, "products must match"):
            CustomerSpec(environment)

    def test_rejects_invalid_budget_and_substitution_values(self):
        environment = load_environment_09()
        environment["base_game"]["customer"]["starting_budget"] = -1
        with self.assertRaisesRegex(ValueError, "starting_budget"):
            CustomerSpec(environment)

        environment = load_environment_09()
        environment["customer_preference"]["substitution_soup_equivalents"][
            "C"
        ] = 0
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            CustomerSpec(environment)

    def test_only_accepts_json_shaped_constructor_input(self):
        valid = CustomerSpec(load_environment_09())

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            CustomerSpec(valid)
        with self.assertRaises(TypeError):
            CustomerSpec(
                starting_budget=valid.starting_budget,
                private_values=valid.private_values,
                minimum_products_per_production_round=(
                    valid.minimum_products_per_production_round
                ),
                substitution_rules=valid.substitution_rules,
            )


if __name__ == "__main__":
    unittest.main()
