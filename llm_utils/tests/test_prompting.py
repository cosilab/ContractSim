from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_utils.prompting import (
    CONTRACT_BEGIN,
    CONTRACT_END,
    PromptRenderError,
    delimit_contract,
    format_system_prompt,
    render_template,
)


class PromptRenderingTests(unittest.TestCase):
    def test_renders_repeated_tokens_and_strips_outer_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompt.txt"
            path.write_text("  {{VALUE}} / {{VALUE}} / {{COUNT}}\n", encoding="utf-8")
            rendered = render_template(path, {"VALUE": "pizza", "COUNT": 2})
        self.assertEqual(rendered, "pizza / pizza / 2")

    def test_replacement_values_are_opaque_and_may_contain_double_braces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompt.txt"
            path.write_text("Contract: {{VALUE}}", encoding="utf-8")
            rendered = render_template(path, {"VALUE": "literal {{contract}}"})
        self.assertEqual(rendered, "Contract: literal {{contract}}")

    def test_missing_tokens_are_reported_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompt.txt"
            path.write_text("{{ZED}} {{ALPHA}}", encoding="utf-8")
            with self.assertRaisesRegex(
                PromptRenderError,
                r"\['ALPHA', 'ZED'\]",
            ):
                render_template(path, {})

    def test_rejects_all_remaining_template_delimiters(self) -> None:
        for template in ("{{lowercase}}", "{{BROKEN", "BROKEN}}"):
            with self.subTest(template=template), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "prompt.txt"
                path.write_text(template, encoding="utf-8")
                with self.assertRaisesRegex(PromptRenderError, "malformed"):
                    render_template(path, {"VALUE": "x"})

    def test_missing_template_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.txt"
            with self.assertRaisesRegex(PromptRenderError, str(path)):
                render_template(path, {})

    def test_template_variables_must_be_a_mapping(self) -> None:
        with self.assertRaisesRegex(PromptRenderError, "mapping"):
            render_template("unused.txt", [])  # type: ignore[arg-type]

    def test_contract_delimiters_preserve_internal_text_and_trim_edges(self) -> None:
        self.assertEqual(
            delimit_contract("  Pizza: $10\nRound 1: one pizza\n"),
            f"{CONTRACT_BEGIN}\nPizza: $10\nRound 1: one pizza\n{CONTRACT_END}",
        )

    def test_system_prompt_has_exact_component_layout(self) -> None:
        self.assertEqual(
            format_system_prompt("maximize utility", "private budget=200"),
            "Goal:\n\nmaximize utility\n\nPrivate Info:\n\nprivate budget=200",
        )

    def test_headers_are_reflowed_outside_but_not_inside_contracts(self) -> None:
        contract = delimit_contract(
            "OBJECTIVES: this is literal contract text\nSCHEDULE: remains literal"
        )
        formatted = format_system_prompt(
            "OBJECTIVES: maximize utility IMPORTANT: obey the protocol",
            f"RECIPES: public recipes\n{contract}\nSCHEDULE: private schedule",
        )
        self.assertIn("OBJECTIVES:\nmaximize utility\n\nIMPORTANT:\nobey", formatted)
        self.assertIn(
            "OBJECTIVES: this is literal contract text\nSCHEDULE: remains literal",
            formatted,
        )
        self.assertIn(f"{CONTRACT_END}\nSCHEDULE:\nprivate schedule", formatted)

    def test_unclosed_contract_boundary_is_preserved(self) -> None:
        text = f"OBJECTIVES: before {CONTRACT_BEGIN}\nIMPORTANT: literal"
        formatted = format_system_prompt(text, "private")
        self.assertIn(f"{CONTRACT_BEGIN}\nIMPORTANT: literal", formatted)


if __name__ == "__main__":
    unittest.main()
