import copy
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment.probability_dist import ProbabilityDist, build_probability_dist
from tests.fixtures import load_environment_09


class ProbabilityDistTests(unittest.TestCase):
    def test_parses_environment_09_as_declarative_distribution_specs(self):
        probability_dist = ProbabilityDist(load_environment_09())

        self.assertEqual(probability_dist.kernel_id, "operating-baseline-v1")
        self.assertEqual(probability_dist.resource_names, ("X", "Y", "Z"))
        self.assertEqual(
            probability_dist.delivery_distribution.distribution,
            "discrete_uniform",
        )
        self.assertEqual(
            probability_dist.delivery_distribution.minimum_received,
            "floor(ordered/2)",
        )
        self.assertTrue(
            probability_dist.delivery_distribution.support_is_inclusive
        )
        self.assertTrue(
            probability_dist.spoilage_distribution.independent_across_resources
        )
        self.assertEqual(
            [
                outcome.units_lost
                for outcome in probability_dist.spoilage_distribution.outcomes
            ],
            [0, 1, 2],
        )
        self.assertEqual(
            probability_dist.price_distribution.resource_names,
            ("X", "Y", "Z"),
        )
        self.assertEqual(
            [
                outcome.value
                for outcome in probability_dist.price_distribution.outcomes_by_resource[
                    "X"
                ]
            ],
            [1.0, 3.0],
        )

    def test_contains_no_compiled_probability_tables(self):
        probability_dist = build_probability_dist(load_environment_09())

        for historical_array in (
            "delivery_table",
            "spoilage_table",
            "price_scenarios",
            "price_probabilities",
        ):
            self.assertFalse(hasattr(probability_dist, historical_array))

    def test_declarative_specs_are_immutable_and_defensive(self):
        source = load_environment_09()
        probability_dist = ProbabilityDist(source)

        source["operating_conditions"]["prices"]["ingredients"]["X"][
            "outcomes"
        ][0]["value"] = 99

        self.assertEqual(
            probability_dist.price_distribution.outcomes_by_resource["X"][0].value,
            1.0,
        )
        with self.assertRaises(TypeError):
            probability_dist.price_distribution.outcomes_by_resource["X"] = ()
        with self.assertRaises(FrozenInstanceError):
            probability_dist.delivery_distribution.distribution = "changed"

    def test_rejects_invalid_environment_probabilities(self):
        environment = copy.deepcopy(load_environment_09())
        environment["operating_conditions"]["spoilage"]["outcomes"][0][
            "probability"
        ] = 0.5

        with self.assertRaisesRegex(ValueError, "must sum to one"):
            build_probability_dist(environment)

    def test_only_accepts_json_shaped_constructor_input(self):
        valid = ProbabilityDist(load_environment_09())

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            ProbabilityDist(valid)
        with self.assertRaises(TypeError):
            ProbabilityDist(
                delivery_distribution=valid.delivery_distribution,
                spoilage_distribution=valid.spoilage_distribution,
                price_distribution=valid.price_distribution,
                resource_names=valid.resource_names,
                kernel_id=valid.kernel_id,
            )


if __name__ == "__main__":
    unittest.main()
