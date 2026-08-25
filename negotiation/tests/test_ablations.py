from __future__ import annotations

import unittest

from negotiation.prompting.ablations import (
    GOAL_TREATMENT_CHOICES,
    TREATMENTS,
    append_goal_treatment,
    get_treatment,
    treatment_metadata,
)


class GoalAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.players = [
            {"name": "customer", "info": "customer info", "goal": "customer goal"},
            {"name": "caterer", "info": "caterer info", "goal": "caterer goal"},
        ]

    def test_control_does_not_modify_players(self) -> None:
        rendered = append_goal_treatment(self.players, "control")
        self.assertEqual(rendered, self.players)
        self.assertIsNot(rendered, self.players)
        self.assertIsNot(rendered[0], self.players[0])

    def test_named_ablation_appends_one_role_specific_clause(self) -> None:
        rendered = append_goal_treatment(self.players, "contingencies")
        for player in rendered:
            role = player["name"]
            clause = TREATMENTS["contingencies"][role]
            self.assertEqual(player["goal"].count(clause), 1)
            self.assertTrue(player["goal"].endswith(clause))
            self.assertNotIn(clause, player["info"])

    def test_unknown_treatment_and_role_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_treatment("unknown")
        with self.assertRaises(ValueError):
            append_goal_treatment(
                [{"name": "observer", "info": "info", "goal": "goal"}],
                "control",
            )

    def test_metadata_matches_public_choices(self) -> None:
        self.assertEqual(
            GOAL_TREATMENT_CHOICES,
            ("control", "contingencies", "net-utility", "no-worse-prior"),
        )
        self.assertEqual(treatment_metadata("control")["goal_clauses"], {})
        self.assertEqual(
            treatment_metadata("net-utility")["goal_clauses"],
            TREATMENTS["net-utility"],
        )


if __name__ == "__main__":
    unittest.main()
