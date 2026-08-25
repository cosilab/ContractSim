from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llm_utils.artifacts import PRIVATE_FILE_MODE
from llm_utils.llm import TraceRecorder


class TraceRecorderTests(unittest.TestCase):
    def test_reset_and_record_create_private_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data" / "trace.jsonl"
            recorder = TraceRecorder(path, enabled=True)
            recorder.reset()
            recorder.record(
                "customer",
                SimpleNamespace(
                    provider="openai",
                    model_name="gpt-test",
                    last_thought_trace="private reasoning",
                    last_prompt_cache_usage=None,
                    last_response_diagnostics=None,
                    chat_metadata={"message_count": 2},
                ),
                '{"response":"hello"}',
                context={"phase": "negotiation", "round": 1},
            )

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["agent"], "customer")
        self.assertEqual(records[0]["reasoning_trace"], "private reasoning")
        self.assertEqual(mode, PRIVATE_FILE_MODE)

    def test_disabled_recorder_does_not_touch_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            recorder = TraceRecorder(path, enabled=False)

            recorder.reset()
            recorder.record("customer", object(), "ignored")

            self.assertEqual(path.read_text(encoding="utf-8"), "existing\n")

    def test_fdopen_failure_closes_the_raw_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            recorder = TraceRecorder(path, enabled=True)
            with (
                mock.patch("llm_utils.llm.tracing.os.open", return_value=91),
                mock.patch("llm_utils.llm.tracing.os.fchmod"),
                mock.patch(
                    "llm_utils.llm.tracing.os.fdopen",
                    side_effect=OSError("fdopen failed"),
                ),
                mock.patch("llm_utils.llm.tracing.os.close") as close,
                self.assertRaisesRegex(OSError, "fdopen failed"),
            ):
                recorder.record("customer", object(), "raw")

        close.assert_called_once_with(91)

    def test_malformed_fields_are_quarantined_without_breaking_jsonl(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            recorder = TraceRecorder(path, enabled=True)
            recorder.record(
                "customer",
                SimpleNamespace(
                    provider="openai",
                    model_name="gpt-test",
                    last_thought_trace=None,
                    last_prompt_cache_usage={"ratio": float("nan")},
                    last_response_diagnostics=object(),
                    chat_metadata={"cycle": cycle},
                ),
                "lone surrogate: \ud800",
                parsed_json={"opaque": object()},
                context={"cycle": cycle},
            )
            recorder.record(
                "caterer",
                SimpleNamespace(provider="anthropic", model_name="claude-test"),
                "normal",
            )

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["raw_output"], "lone surrogate: \ud800")
        self.assertIsNone(records[0]["context"])
        self.assertIsNone(records[0]["parsed_json"])
        self.assertIsNone(records[0]["prompt_cache"])
        self.assertIsNone(records[0]["response_diagnostics"])
        self.assertIsNone(records[0]["chat"])
        self.assertEqual(
            records[0]["serialization_errors"],
            {
                "chat": "RecursionError",
                "context": "RecursionError",
                "parsed_json": "TypeError",
                "prompt_cache": "ValueError",
                "response_diagnostics": "TypeError",
            },
        )
        self.assertEqual(records[1]["raw_output"], "normal")


if __name__ == "__main__":
    unittest.main()
