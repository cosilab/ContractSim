import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment.supplier import SupplierSpec
from tests.fixtures import load_environment_09


class SupplierSpecTests(unittest.TestCase):
    def test_loads_the_environment_shape_used_by_the_current_solver(self):
        spec = SupplierSpec(load_environment_09())

        self.assertEqual(spec.starting_capital, 20)
        self.assertEqual(spec.number_of_capital_states, 221)
        self.assertEqual(spec.max_inventory_per_resource, 10)
        self.assertEqual(spec.max_order_per_resource_per_round, 12)
        self.assertEqual(spec.products, ("A", "B", "C"))
        self.assertEqual(spec.resources, ("X", "Y", "Z"))
        self.assertEqual(dict(spec.recipes["A"]), {"X": 1, "Y": 1, "Z": 1})

    def test_is_immutable_and_defensively_copies_nested_recipes(self):
        environment = load_environment_09()
        spec = SupplierSpec(environment)

        environment["base_game"]["recipes"]["A"]["X"] = 99

        self.assertEqual(spec.recipes["A"]["X"], 1)
        with self.assertRaises(TypeError):
            spec.recipes["A"]["X"] = 99
        with self.assertRaises(FrozenInstanceError):
            spec.starting_capital = 0

    def test_rejects_capital_outside_the_discrete_state_space(self):
        environment = load_environment_09()
        environment["caterer_capital"]["starting_balance"] = 221

        with self.assertRaisesRegex(ValueError, "starting_capital must be smaller"):
            SupplierSpec(environment)

    def test_rejects_empty_or_invalid_recipes(self):
        environment = load_environment_09()
        environment["base_game"]["recipes"]["A"] = {"X": 0}

        with self.assertRaisesRegex(ValueError, "at least one resource"):
            SupplierSpec(environment)

        environment = load_environment_09()
        environment["base_game"]["recipes"]["A"]["X"] = 1.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            SupplierSpec(environment)

    def test_only_accepts_json_shaped_constructor_input(self):
        valid = SupplierSpec(load_environment_09())

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            SupplierSpec(valid)
        with self.assertRaises(TypeError):
            SupplierSpec(
                starting_capital=valid.starting_capital,
                number_of_capital_states=valid.number_of_capital_states,
                recipes=valid.recipes,
                max_inventory_per_resource=valid.max_inventory_per_resource,
                max_order_per_resource_per_round=(
                    valid.max_order_per_resource_per_round
                ),
            )


if __name__ == "__main__":
    unittest.main()
