from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_utils.llm.config import RuntimeOptions, load_provider_config, resolve_api_key


class SharedProviderConfigTests(unittest.TestCase):
    def test_loads_defaults_and_unprefixed_overrides(self) -> None:
        defaults = load_provider_config("openai", environ={})
        self.assertEqual(defaults.model, "gpt-5.6-sol")
        self.assertEqual(defaults.reasoning_metadata, {"effort": "high"})

        overridden = load_provider_config(
            "gemini",
            environ={
                "GEMINI_MODEL": "gemini-test",
                "GEMINI_THINKING_LEVEL": "medium",
            },
        )
        self.assertEqual(overridden.model, "gemini-test")
        self.assertEqual(
            overridden.reasoning_metadata,
            {"thinking_level": "medium"},
        )
        self.assertEqual(
            load_provider_config("OpenAI", environ={}).provider,
            "openai",
        )
        self.assertEqual(RuntimeOptions().cache_scope, "ratconbench")

    def test_api_key_prefers_environment_then_key_file(self) -> None:
        self.assertEqual(
            resolve_api_key("openai", environ={"OPENAI_API_KEY": "from-env"}),
            "from-env",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            key_file = Path(temp_dir) / "ANTHROPIC_API.key"
            key_file.write_text("from-file\n", encoding="utf-8")
            self.assertEqual(
                resolve_api_key("anthropic", keys_dir=temp_dir, environ={}),
                "from-file",
            )

    def test_missing_api_key_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                resolve_api_key("anthropic", keys_dir=temp_dir, environ={})


if __name__ == "__main__":
    unittest.main()
