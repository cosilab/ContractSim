"""Optional structured trace logging for model calls."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..artifacts import PRIVATE_FILE_MODE, json_safe, write_text


def _strict_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    """Quarantine malformed diagnostics without losing the whole trace line."""
    converted: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for field, value in record.items():
        try:
            converted[field] = json_safe(value)
        except (TypeError, ValueError, RecursionError) as exc:
            converted[field] = None
            errors[field] = type(exc).__name__
    if errors:
        converted["serialization_errors"] = errors
    return converted


class TraceRecorder:
    """Append LLM diagnostics to one JSONL file when tracing is enabled."""

    def __init__(self, path: str | Path | None, *, enabled: bool) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = enabled

    def reset(self) -> None:
        if not self.enabled or self.path is None:
            return
        write_text(self.path, "")

    def record(
        self,
        agent_name: str,
        model: Any,
        raw_output: str,
        parser_input: str | None = None,
        parsed_json: Any = None,
        parse_meta: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.path is None:
            return
        context = context or {}
        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent_name,
            "provider": getattr(model, "provider", None),
            "model": getattr(model, "model_name", None),
            "phase": context.get("phase"),
            "action": context.get("action"),
            "round": context.get("round"),
            "accepted": context.get("accepted"),
            "call_to_action": context.get("call_to_action"),
            "context": context,
            "raw_output": raw_output,
            "parser_input": parser_input if parser_input is not None else raw_output,
            "parse": parse_meta or {},
            "parsed_json": parsed_json,
            "reasoning_trace": getattr(model, "last_thought_trace", None),
            "prompt_cache": getattr(model, "last_prompt_cache_usage", None),
            "response_diagnostics": getattr(
                model,
                "last_response_diagnostics",
                None,
            ),
            "chat": getattr(model, "chat_metadata", None),
        }
        serialized = json.dumps(
            _strict_trace_record(record),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, PRIVATE_FILE_MODE)
        try:
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
            trace_file = os.fdopen(descriptor, "a", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        with trace_file:
            trace_file.write(serialized)
