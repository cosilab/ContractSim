from __future__ import annotations

import json
import stat
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from llm_utils.artifacts import (
    PRIVATE_FILE_MODE,
    PUBLIC_FILE_MODE,
    RunClaimError,
    claim_output_directory,
    json_safe,
    write_json,
    write_text,
)


@dataclass(frozen=True)
class _Record:
    path: Path
    values: tuple[int, ...]


class ArtifactSerializationTests(unittest.TestCase):
    def test_json_safe_converts_supported_values_recursively(self) -> None:
        self.assertEqual(
            json_safe(
                {
                    "record": _Record(Path("café.txt"), (1, 2)),
                    7: [True, None, 1.5],
                }
            ),
            {
                "record": {"path": "café.txt", "values": [1, 2]},
                "7": [True, None, 1.5],
            },
        )

    def test_json_safe_rejects_opaque_and_non_finite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                json_safe({"value": value})
        with self.assertRaises(TypeError):
            json_safe({"value": object()})

    def test_atomic_writes_are_utf8_strict_and_mode_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private_path = write_text(root / "private.txt", "café")
            public_path = write_json(
                root / "public.json",
                {"message": "你好"},
                mode=PUBLIC_FILE_MODE,
            )

            self.assertEqual(private_path.read_text(encoding="utf-8"), "café")
            self.assertEqual(
                json.loads(public_path.read_text(encoding="utf-8")),
                {"message": "你好"},
            )
            self.assertEqual(
                stat.S_IMODE(private_path.stat().st_mode),
                PRIVATE_FILE_MODE,
            )
            self.assertEqual(
                stat.S_IMODE(public_path.stat().st_mode),
                PUBLIC_FILE_MODE,
            )
            self.assertTrue(public_path.read_text(encoding="utf-8").endswith("\n"))

    def test_atomic_replacement_tightens_existing_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.txt"
            path.write_text("old", encoding="utf-8")
            path.chmod(PUBLIC_FILE_MODE)

            write_text(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "new")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), PRIVATE_FILE_MODE)
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp-*")), [])

    def test_failed_atomic_replace_preserves_destination_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.txt"
            path.write_text("original", encoding="utf-8")
            with (
                mock.patch(
                    "llm_utils.artifacts.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                write_text(path, "replacement")

            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp-*")), [])

    def test_text_and_mode_inputs_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.txt"
            with self.assertRaises(TypeError):
                write_text(path, b"bytes")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                write_text(path, "text", mode=True)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                write_text(path, "text", mode=0o1000)


class RunClaimTests(unittest.TestCase):
    def test_claim_allows_scheduler_files_and_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            scheduler_log = output / "stdout.log"
            scheduler_log.write_text("scheduler output", encoding="utf-8")

            claim = claim_output_directory(output, metadata={"job_id": "job-1"})

            status = json.loads(claim.status_path.read_text(encoding="utf-8"))
            self.assertEqual(
                status,
                {"schema_version": 1, "status": "running", "job_id": "job-1"},
            )
            self.assertEqual(
                scheduler_log.read_text(encoding="utf-8"),
                "scheduler output",
            )
            self.assertEqual(
                stat.S_IMODE(claim.status_path.stat().st_mode),
                PRIVATE_FILE_MODE,
            )
            with self.assertRaisesRegex(RunClaimError, "claimed by another run"):
                claim_output_directory(output)

    def test_normal_context_exit_completes_an_unfinalized_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with claim_output_directory(
                Path(temporary) / "run",
                metadata={"job_id": "job-1"},
            ) as claim:
                self.assertFalse(claim.finalized)

            self.assertTrue(claim.finalized)
            self.assertEqual(claim.terminal_status, "complete")
            self.assertEqual(
                json.loads(claim.status_path.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "status": "complete",
                    "job_id": "job-1",
                },
            )

    def test_explicit_completion_supports_payload_and_custom_terminal_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            claim = claim_output_directory(Path(temporary) / "run")
            claim.finalize(
                "invalid",
                {"schema_version": 1, "status": "invalid", "valid": False},
            )

            self.assertEqual(
                json.loads(claim.status_path.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "status": "invalid",
                    "valid": False,
                },
            )
            with self.assertRaisesRegex(RunClaimError, "already finalized"):
                claim.complete()

    def test_exceptional_context_exit_records_error_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(ValueError, "post-claim failure"):
                with claim_output_directory(
                    output,
                    metadata={"job_id": "job-1"},
                ):
                    raise ValueError("post-claim failure")

            status_path = output / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["job_id"], "job-1")
            self.assertEqual(
                status["error"],
                {"type": "ValueError", "message": "post-claim failure"},
            )

    def test_exception_after_explicit_completion_overwrites_it_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaisesRegex(RuntimeError, "late failure"):
                with claim_output_directory(output) as claim:
                    claim.complete({"valid": True})
                    raise RuntimeError("late failure")

            status = json.loads(
                (output / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["error"]["type"], "RuntimeError")

    def test_invalid_metadata_does_not_create_the_claim_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with self.assertRaises(ValueError):
                claim_output_directory(output, metadata={"value": float("nan")})
            self.assertFalse(output.exists())
            self.assertFalse((output / "run_status.json").exists())

    def test_error_payload_must_be_a_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            claim = claim_output_directory(Path(temporary) / "run")
            with self.assertRaises(TypeError):
                claim.fail("failed", [])  # type: ignore[arg-type]
            self.assertEqual(
                json.loads(claim.status_path.read_text(encoding="utf-8"))["status"],
                "running",
            )

    def test_claim_validates_paths_and_can_be_public(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file_path = root / "file"
            file_path.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(RunClaimError, "not a directory"):
                claim_output_directory(file_path)
            with self.assertRaises(ValueError):
                claim_output_directory(root / "run", status_filename="../status.json")

            claim = claim_output_directory(
                root / "public-run",
                mode=PUBLIC_FILE_MODE,
            )
            claim.complete()
            self.assertEqual(
                stat.S_IMODE(claim.status_path.stat().st_mode),
                PUBLIC_FILE_MODE,
            )

    def test_failed_permission_setup_removes_a_partial_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with (
                mock.patch(
                    "llm_utils.artifacts.os.fchmod",
                    side_effect=OSError("chmod failed"),
                ),
                self.assertRaisesRegex(OSError, "chmod failed"),
            ):
                claim_output_directory(output)
            self.assertFalse((output / "run_status.json").exists())


if __name__ == "__main__":
    unittest.main()
