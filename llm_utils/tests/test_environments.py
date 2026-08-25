from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_utils.environments import (
    DEFAULT_ENVIRONMENT_ROOT,
    EnvironmentLoadError,
    load_environment,
    load_environment_metadata,
    load_prompt_context,
)


class EnvironmentLoadingTests(unittest.TestCase):
    def test_default_root_is_repository_relative_and_cwd_independent(self) -> None:
        expected = Path(__file__).resolve().parents[2] / "envs"
        self.assertEqual(DEFAULT_ENVIRONMENT_ROOT, expected)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                self.assertEqual(load_environment(1)["environment_id"], 1)
            finally:
                os.chdir(previous)

    def test_all_canonical_snapshots_and_contexts_have_matching_identity(self) -> None:
        for environment_id in range(1, 7):
            with self.subTest(environment_id=environment_id):
                environment = load_environment(environment_id)
                context = load_prompt_context(environment_id)
                metadata = load_environment_metadata(environment_id)
                self.assertEqual(environment["environment_id"], environment_id)
                self.assertEqual(context["environment_id"], environment_id)
                self.assertEqual(
                    metadata,
                    {
                        "environment_id": environment_id,
                        "environment_name": environment["environment_name"],
                        "environment_axes": environment["axes"],
                    },
                )
                self.assertIsInstance(context["template_variables"], dict)

    def test_custom_root_loads_fresh_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            environment_dir = root / "8"
            environment_dir.mkdir()
            (environment_dir / "environment.json").write_text(
                json.dumps(
                    {
                        "environment_id": 8,
                        "environment_name": "test-environment",
                        "axes": {"risk": "low"},
                        "payload": [1, 2, 3],
                    }
                ),
                encoding="utf-8",
            )
            (environment_dir / "prompt-context.json").write_text(
                json.dumps(
                    {
                        "environment_id": 8,
                        "template_variables": {"VALUE": 1},
                    }
                ),
                encoding="utf-8",
            )

            first = load_environment(8, root)
            first["payload"].append(4)
            second = load_environment(8, root)

            self.assertEqual(second["payload"], [1, 2, 3])
            self.assertEqual(load_prompt_context(8, root)["template_variables"], {"VALUE": 1})
            self.assertEqual(
                load_environment_metadata(8, root)["environment_axes"],
                {"risk": "low"},
            )

    def test_rejects_invalid_environment_ids(self) -> None:
        for invalid in (True, False, "1", 1.0, 0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(EnvironmentLoadError):
                    load_environment(invalid)  # type: ignore[arg-type]

    def test_reports_missing_malformed_and_non_object_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(EnvironmentLoadError, "Missing"):
                load_environment(1, root)

            environment_dir = root / "1"
            environment_dir.mkdir()
            path = environment_dir / "environment.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(EnvironmentLoadError, "Invalid JSON"):
                load_environment(1, root)

            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(EnvironmentLoadError, "JSON object"):
                load_environment(1, root)

    def test_rejects_document_identity_mismatches(self) -> None:
        for declared_id in (3, True, "2", 2.0, None):
            with (
                self.subTest(declared_id=declared_id),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                environment_dir = root / "2"
                environment_dir.mkdir()
                (environment_dir / "environment.json").write_text(
                    json.dumps({"environment_id": declared_id}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(EnvironmentLoadError, "does not match"):
                    load_environment(2, root)

    def test_validates_prompt_context_shape(self) -> None:
        for variables in (None, [], "variables"):
            with self.subTest(variables=variables), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                environment_dir = root / "1"
                environment_dir.mkdir()
                (environment_dir / "prompt-context.json").write_text(
                    json.dumps(
                        {
                            "environment_id": 1,
                            "template_variables": variables,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(EnvironmentLoadError, "template_variables"):
                    load_prompt_context(1, root)

    def test_validates_metadata_shape(self) -> None:
        invalid_documents = (
            {"environment_id": 1, "environment_name": "", "axes": {}},
            {"environment_id": 1, "environment_name": 7, "axes": {}},
            {"environment_id": 1, "environment_name": "valid", "axes": []},
            {
                "environment_id": 1,
                "environment_name": "valid",
                "axes": {"risk": 1},
            },
        )
        for document in invalid_documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                environment_dir = root / "1"
                environment_dir.mkdir()
                (environment_dir / "environment.json").write_text(
                    json.dumps(document),
                    encoding="utf-8",
                )
                with self.assertRaises(EnvironmentLoadError):
                    load_environment_metadata(1, root)


if __name__ == "__main__":
    unittest.main()
