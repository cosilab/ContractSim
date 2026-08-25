from __future__ import annotations

import unittest

from performance.engine.environment import (
    EnvironmentEngine,
    load_performance_environment,
)


class PerformanceEnvironmentTests(unittest.TestCase):
    def test_all_six_environments_load_the_execution_fields(self) -> None:
        environments = [
            load_performance_environment(environment_id)
            for environment_id in range(1, 7)
        ]

        self.assertEqual(
            [environment.environment_id for environment in environments],
            list(range(1, 7)),
        )
        self.assertTrue(
            all(environment.total_weeks == 11 for environment in environments)
        )
        self.assertTrue(
            all(
                environment.payment_weeks == (1, 3, 5, 7, 9, 11)
                and environment.production_weeks == (2, 4, 6, 8, 10)
                for environment in environments
            )
        )
        self.assertEqual(
            [environment.supplier_starting_balance for environment in environments],
            [20, 40, 20, 40, 20, 40],
        )
        self.assertEqual(
            [environment.minimum_received(7) for environment in environments],
            [7, 7, 5, 5, 3, 3],
        )
        self.assertEqual(
            [environment.resample_prices for environment in environments],
            [False, False, True, True, True, True],
        )

    def test_seeded_stochastic_transitions_are_reproducible(self) -> None:
        for environment_id in range(1, 7):
            with self.subTest(environment_id=environment_id):
                first = EnvironmentEngine(environment_id, seed=123)
                second = EnvironmentEngine(environment_id, seed=123)
                first.apply_payment(20)
                second.apply_payment(20)

                first_purchase = first.apply_purchase({"X": 2, "Y": 2, "Z": 2})
                second_purchase = second.apply_purchase({"X": 2, "Y": 2, "Z": 2})
                self.assertEqual(first_purchase, second_purchase)
                self.assertEqual(first.inventory, second.inventory)

                self.assertEqual(first.apply_spoilage(), second.apply_spoilage())
                first.roll_prices()
                second.roll_prices()
                self.assertEqual(first.ingredient_prices, second.ingredient_prices)

    def test_payment_purchase_and_production_update_state(self) -> None:
        game = EnvironmentEngine(1, seed=42)
        game.apply_payment(10)
        self.assertEqual(game.customer_budget, 190)
        self.assertEqual(game.supplier_balance, 30)

        cost, ordered, received = game.apply_purchase(
            {"X": 2, "Y": 2, "Z": 2}
        )
        self.assertEqual(cost, 8)
        self.assertEqual(ordered, {"X": 2, "Y": 2, "Z": 2})
        self.assertEqual(received, ordered)

        produced = game.apply_production({"A": 1, "B": 1, "C": 1})
        self.assertEqual(produced, {"A": 1, "B": 1, "C": 1})
        self.assertEqual(game.inventory, {"X": 0, "Y": 0, "Z": 0})
        self.assertEqual(game.delivered, produced)

    def test_invalid_actions_are_rejected(self) -> None:
        game = EnvironmentEngine(1, seed=42)
        with self.assertRaisesRegex(ValueError, "customer's budget"):
            game.apply_payment(201)
        with self.assertRaisesRegex(ValueError, "order limit"):
            game.apply_purchase({"X": 13, "Y": 0, "Z": 0})
        with self.assertRaisesRegex(ValueError, "available inventory"):
            game.apply_production({"A": 1, "B": 0, "C": 0})


if __name__ == "__main__":
    unittest.main()
