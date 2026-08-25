"""Parse the structured responses emitted during negotiation."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _json_error(exc: json.JSONDecodeError) -> dict[str, Any]:
    return {
        "message": exc.msg,
        "line": exc.lineno,
        "column": exc.colno,
        "position": exc.pos,
    }


def _normalize_contract(contract: str | None) -> str | None:
    if contract is None:
        return None
    normalized = contract.replace("\\n", "\n").strip()
    return normalized or None


def parse_negotiation_response(
    raw: str,
) -> tuple[str | None, str | None, bool, bool, dict[str, Any]]:
    """Parse dialogue, contract, acceptance and termination with diagnostics."""
    cleaned = _JSON_FENCE_RE.sub("", raw).strip()
    meta: dict[str, Any] = {
        "parser": "negotiation_json",
        "json_input": cleaned,
        "parsed": False,
        "method": None,
    }
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        meta.update({"method": "invalid_json", "error": _json_error(exc)})
        return None, None, False, False, meta

    meta.update(
        {"parsed": True, "method": "whole_output", "parsed_json": parsed}
    )

    if isinstance(parsed, dict) and "response" in parsed:
        response = parsed.get("response")
        raw_contract = parsed.get("contract")
        invalid_fields = []
        if not isinstance(response, str):
            invalid_fields.append("response must be a string")
        if raw_contract is not None and not isinstance(raw_contract, str):
            invalid_fields.append("contract must be a string or null")
        if invalid_fields:
            meta.update(
                {
                    "schema_valid": False,
                    "invalid_fields": invalid_fields,
                    "parsed_fields": {
                        "response": None,
                        "contract": None,
                        "accept": False,
                        "terminate": False,
                    },
                }
            )
            return None, None, False, False, meta

        contract = _normalize_contract(raw_contract)
        accept = parsed.get("accept") is True
        terminate = parsed.get("terminate") is True
        meta.update(
            {
                "schema_valid": True,
                "parsed_fields": {
                    "response": response,
                    "contract": contract,
                    "accept": accept,
                    "terminate": terminate,
                },
            }
        )
        return response, contract, accept, terminate, meta

    meta.update(
        {
            "schema_valid": False,
            "invalid_fields": ["expected an object containing a response field"],
            "parsed_fields": {
                "response": None,
                "contract": None,
                "accept": False,
                "terminate": False,
            },
        }
    )
    return None, None, False, False, meta


def strip_name_prefix(name: str, text: str) -> str:
    for prefix in (f"{name}:", f"{name} "):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text
