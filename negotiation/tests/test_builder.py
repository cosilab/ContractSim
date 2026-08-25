from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from negotiation.prompting import (
    PromptBuildError,
    TREATMENTS,
    build_prompts,
)


class PromptBuilderTests(unittest.TestCase):
    def test_builds_both_roles_without_unresolved_tokens(self) -> None:
        players = build_prompts(5)

        self.assertEqual([player["name"] for player in players], ["customer", "caterer"])
        for player in players:
            self.assertNotIn("{{", player["info"] + player["goal"])
            self.assertNotIn("}}", player["info"] + player["goal"])

    def test_negotiation_horizon_uses_supplied_max_rounds_for_both_roles(self) -> None:
        players = build_prompts(5, max_rounds=7)

        for player in players:
            self.assertIn("at most 7 rounds", player["goal"])
            self.assertIn("within 7 rounds", player["goal"])
            self.assertNotIn("50 rounds", player["goal"])

    def test_max_rounds_must_be_a_positive_integer(self) -> None:
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(PromptBuildError, "positive integer"):
                    build_prompts(1, max_rounds=invalid)

    def test_environment_changes_only_the_expected_role(self) -> None:
        environment_1 = {player["name"]: player for player in build_prompts(1)}
        environment_3 = {player["name"]: player for player in build_prompts(3)}

        self.assertEqual(environment_1["customer"], environment_3["customer"])
        self.assertNotEqual(
            environment_1["caterer"]["info"],
            environment_3["caterer"]["info"],
        )

    def test_goal_treatment_changes_only_goal_components(self) -> None:
        control = {player["name"]: player for player in build_prompts(5)}
        treated = {
            player["name"]: player
            for player in build_prompts(5, goal_treatment="contingencies")
        }

        treatment = TREATMENTS["contingencies"]
        for role in ("customer", "caterer"):
            self.assertEqual(control[role]["info"], treated[role]["info"])
            self.assertEqual(
                treated[role]["goal"],
                control[role]["goal"].rstrip()
                + "\n\n"
                + treatment[role],
            )

    def test_accepts_another_domain_prompt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for role in ("customer", "caterer"):
                role_dir = root / role
                role_dir.mkdir(parents=True)
                (role_dir / "info.txt").write_text(
                    f"{role} budget={{{{CUSTOMER_BUDGET}}}}",
                    encoding="utf-8",
                )
                (role_dir / "goal.txt").write_text(
                    f"{role} goal",
                    encoding="utf-8",
                )

            players = build_prompts(1, domain_prompt_root=root)

        self.assertEqual(players[0]["info"], "customer budget=200")
        self.assertEqual(players[1]["info"], "caterer budget=200")

if __name__ == "__main__":
    unittest.main()
