from __future__ import annotations

import unittest

from llm_utils import load_prompt_context
from llm_utils.prompting import CONTRACT_BEGIN, CONTRACT_END
from performance.prompting import PromptBuildError, TREATMENTS, build_prompts


class PerformancePromptTests(unittest.TestCase):
    def test_all_six_environments_render_role_specific_prompts(self) -> None:
        for environment_id in range(1, 7):
            with self.subTest(environment_id=environment_id):
                players = build_prompts(environment_id, "Price: $10")
                by_role = {player["name"]: player for player in players}
                context = load_prompt_context(environment_id)[
                    "template_variables"
                ]

                self.assertEqual(set(by_role), {"customer", "supplier"})
                self.assertIn(
                    context["CUSTOMER_PRIVATE_VALUES_PERFORMANCE"],
                    by_role["customer"]["info"],
                )
                self.assertIn(
                    context["CUSTOMER_MINIMUM_REQUIREMENTS"],
                    by_role["customer"]["info"],
                )
                self.assertIn(
                    context["OPERATING_CONDITIONS"],
                    by_role["supplier"]["info"],
                )
                self.assertIn(
                    f"${context['CATERER_CAPITAL']}",
                    by_role["supplier"]["info"],
                )
                for player in players:
                    self.assertIn(CONTRACT_BEGIN, player["info"])
                    self.assertIn(CONTRACT_END, player["info"])
                    self.assertNotIn("{{", player["info"] + player["goal"])

    def test_goal_treatment_changes_only_the_selected_role_goal(self) -> None:
        control = {
            player["name"]: player
            for player in build_prompts(1, "Contract", goal_treatment="control")
        }
        treated = {
            player["name"]: player
            for player in build_prompts(
                1,
                "Contract",
                goal_treatment="plan-ahead",
                treated_role="supplier",
            )
        }

        self.assertEqual(control["customer"], treated["customer"])
        self.assertEqual(control["supplier"]["info"], treated["supplier"]["info"])
        self.assertEqual(
            treated["supplier"]["goal"],
            control["supplier"]["goal"]
            + "\n\n"
            + TREATMENTS["plan-ahead"].clause,
        )

    def test_empty_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(PromptBuildError, "must be non-empty"):
            build_prompts(1, "   ")


if __name__ == "__main__":
    unittest.main()
