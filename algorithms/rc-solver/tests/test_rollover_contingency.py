import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contingencies import RolloverContingency
from environment import Environment
from tests.fixtures import load_environment_09


class RolloverContingencyTests(unittest.TestCase):
    def setUp(self) -> None:
        source = load_environment_09()
        source["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ] = {"A": 0, "B": 0, "C": 0}
        self.environment = Environment(source)
        self.rollover = RolloverContingency(self.environment, max_deficit=2)
        self.zero_deficit = {"A": 0, "B": 0, "C": 0}

    def test_weekly_minimums_do_not_override_smaller_ingredient_use(self):
        source = load_environment_09()
        source["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ] = {"A": 1, "B": 0, "C": 0}
        rollover = RolloverContingency(Environment(source), max_deficit=2)

        outcome = rollover.select_production(
            target_quantities={"A": 1, "B": 0, "C": 2},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 1, "Y": 1, "Z": 2},
        )

        self.assertTrue(outcome.survives)
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 2})
        self.assertEqual(dict(outcome.next_deficit), {"A": 1, "B": 0, "C": 0})

    def test_weekly_minimum_failure_can_roll_over_only_without_prior_deficit(self):
        source = load_environment_09()
        source["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ] = {"A": 1, "B": 0, "C": 0}
        rollover = RolloverContingency(Environment(source), max_deficit=2)

        first = rollover.select_production(
            target_quantities={"A": 2, "B": 0, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 0, "Y": 0, "Z": 0},
        )
        repeated = rollover.select_production(
            target_quantities={"A": 2, "B": 0, "C": 0},
            previous_deficit={"A": 1, "B": 0, "C": 0},
            available_inventory={"X": 0, "Y": 0, "Z": 0},
        )

        self.assertTrue(first.survives)
        self.assertTrue(repeated.terminates)
        self.assertEqual(dict(first.production), self.zero_deficit)
        self.assertEqual(dict(first.next_deficit), {"A": 2, "B": 0, "C": 0})
        self.assertEqual(dict(repeated.production), self.zero_deficit)

    def test_previous_deficit_is_added_to_the_complete_obligation(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 1, "B": 0, "C": 0},
            previous_deficit={"A": 1, "B": 0, "C": 0},
            available_inventory={"X": 2, "Y": 2, "Z": 2},
        )

        self.assertEqual(dict(outcome.effective_requirement), {"A": 2, "B": 0, "C": 0})
        self.assertEqual(dict(outcome.production), {"A": 2, "B": 0, "C": 0})
        self.assertTrue(outcome.complete_fulfilment)
        self.assertEqual(dict(outcome.next_deficit), self.zero_deficit)

    def test_partial_ranking_is_unweighted_then_uses_fewer_ingredients(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 1, "B": 0, "C": 2},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 1, "Y": 1, "Z": 2},
        )

        # A+C has much greater private value, but it serves the same dish count
        # as 2C and uses more ingredients. Private values do not rank plans.
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 2})
        self.assertEqual(outcome.total_shortfall, 1)
        self.assertEqual(sum(outcome.ingredient_usage.values()), 2)

    def test_first_bounded_shortfall_survives_and_is_credited(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 2, "B": 0, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 1, "Y": 1, "Z": 1},
        )

        self.assertTrue(outcome.survives)
        self.assertFalse(outcome.complete_fulfilment)
        self.assertEqual(dict(outcome.production), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(dict(outcome.next_deficit), {"A": 1, "B": 0, "C": 0})
        self.assertEqual(outcome.customer_value, 12)

    def test_repeated_shortfall_invokes_zero_output_grim(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 1, "B": 0, "C": 0},
            previous_deficit={"A": 1, "B": 0, "C": 0},
            available_inventory={"X": 1, "Y": 1, "Z": 1},
        )

        self.assertTrue(outcome.terminates)
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 0})
        self.assertEqual(outcome.customer_value, 0)

    def test_oversized_first_shortfall_invokes_zero_output_grim(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 4, "B": 0, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 1, "Y": 1, "Z": 1},
        )

        self.assertTrue(outcome.terminates)
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 0})
        self.assertEqual(outcome.customer_value, 0)

    def test_final_round_shortfall_invokes_zero_output_grim(self):
        outcome = self.rollover.select_production(
            target_quantities={"A": 2, "B": 0, "C": 0},
            previous_deficit=self.zero_deficit,
            available_inventory={"X": 1, "Y": 1, "Z": 1},
            allow_rollover=False,
        )

        self.assertTrue(outcome.terminates)
        self.assertEqual(dict(outcome.production), {"A": 0, "B": 0, "C": 0})
        self.assertEqual(outcome.customer_value, 0)


if __name__ == "__main__":
    unittest.main()
