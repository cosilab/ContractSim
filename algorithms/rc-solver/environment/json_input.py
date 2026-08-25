"""Shared loading and primitive validation for JSON-shaped environment input."""

from __future__ import annotations

import json
from math import fsum, isclose, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def load_json_object(path: str | Path, *, name: str = "JSON") -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""
    source = json.loads(Path(path).read_text())
    if not isinstance(source, dict):
        raise TypeError(f"{name} must contain an object")
    return source


def require_json_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require an already parsed JSON-style object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON-shaped mapping")
    return value


def require_mapping(
    container: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    """Read a required nested mapping from a JSON object."""
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping")
    return value


def require_finite_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce a JSON number while rejecting booleans and non-finite values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return number


def require_integer(value: Any, *, name: str, minimum: int) -> int:
    """Coerce a bounded JSON integer without silently truncating fractions."""
    requirement = f"{name} must be at least {minimum}; value must be an integer"
    if isinstance(value, bool):
        raise ValueError(requirement)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(requirement) from exc
    if not isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(requirement)
    return int(number)


def require_nonempty_string(value: Any, *, name: str) -> str:
    """Require and trim a non-empty JSON string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def require_probability(value: Any, *, name: str) -> float:
    """Require one finite probability in the closed interval [0, 1]."""
    return require_finite_number(value, name=name, minimum=0.0, maximum=1.0)


def validate_probability_sum(
    probabilities: Sequence[float],
    *,
    name: str,
) -> None:
    """Require a non-empty probability vector that sums to one."""
    if len(probabilities) == 0 or not isclose(
        fsum(probabilities),
        1.0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{name} probabilities must sum to one")


def immutable_integer_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
    minimum: int,
) -> Mapping[str, int]:
    """Validate and defensively freeze a string-to-integer JSON mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        normalized[key] = require_integer(
            raw,
            name=f"{name}[{key!r}]",
            minimum=minimum,
        )
    if not normalized:
        raise ValueError(f"{name} must contain at least one entry")
    return MappingProxyType(normalized)
