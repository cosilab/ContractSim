import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment.json_input import (
    immutable_integer_mapping,
    load_json_object,
    require_finite_number,
    require_integer,
    require_json_mapping,
    require_mapping,
    require_nonempty_string,
    require_probability,
    validate_probability_sum,
)


class JsonInputTests(unittest.TestCase):
    def test_loads_a_json_object_and_rejects_non_object_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            object_path = Path(directory) / "object.json"
            object_path.write_text(json.dumps({"environment_id": 9}))
            self.assertEqual(load_json_object(object_path), {"environment_id": 9})

            array_path = Path(directory) / "array.json"
            array_path.write_text("[]")
            with self.assertRaisesRegex(TypeError, "must contain an object"):
                load_json_object(array_path, name="Environment JSON")

    def test_requires_json_and_nested_mappings(self):
        source = require_json_mapping({"base_game": {"customer": {}}}, name="Input")
        base_game = require_mapping(source, "base_game", path="environment")
        self.assertEqual(
            require_mapping(base_game, "customer", path="environment.base_game"),
            {},
        )

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            require_json_mapping([], name="Input")
        with self.assertRaisesRegex(ValueError, "environment.base_game"):
            require_mapping({}, "base_game", path="environment")

    def test_validates_numbers_integers_and_strings(self):
        self.assertEqual(require_finite_number(2.5, name="value"), 2.5)
        self.assertEqual(require_integer(3.0, name="count", minimum=0), 3)
        self.assertEqual(require_nonempty_string("  current ", name="axis"), "current")

        for invalid in (True, 1.5, -1, float("inf")):
            with self.assertRaises(ValueError):
                require_integer(invalid, name="count", minimum=0)

    def test_validates_individual_and_total_probabilities(self):
        self.assertEqual(require_probability(0.25, name="probability"), 0.25)
        validate_probability_sum([0.25, 0.75], name="outcome")

        for invalid in (-0.1, 1.1, True):
            with self.assertRaises(ValueError):
                require_probability(invalid, name="probability")
        with self.assertRaisesRegex(ValueError, "must sum to one"):
            validate_probability_sum([0.2, 0.2], name="outcome")

    def test_builds_an_immutable_defensive_integer_mapping(self):
        source = {"A": 1, "B": 2}
        result = immutable_integer_mapping(source, name="quantities", minimum=0)
        source["A"] = 99

        self.assertEqual(dict(result), {"A": 1, "B": 2})
        with self.assertRaises(TypeError):
            result["A"] = 99


if __name__ == "__main__":
    unittest.main()
