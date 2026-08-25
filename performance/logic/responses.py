"""Parse structured performance-action responses."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_FLAT_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action_response(
    raw: str,
    required_keys: Sequence[str],
) -> dict[str, Any] | None:
    """Return a JSON action containing every required key."""
    cleaned = _JSON_FENCE_RE.sub("", str(raw)).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _FLAT_OBJECT_RE.search(cleaned)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None
    if any(key not in parsed for key in required_keys):
        return None
    return parsed


def strip_name_prefix(name: str, text: str) -> str:
    for prefix in (f"{name}:", f"{name} "):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text
