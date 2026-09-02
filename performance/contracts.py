"""Synthetic contract metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path


def contract_id(contract_dir: str | Path) -> str:
    """Return the configured stable ID, falling back to the directory name."""
    path = Path(contract_dir)
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return path.name
    except (OSError, json.JSONDecodeError) as exc:
        config_path = path / "config.json"
        raise ValueError(f"Cannot load contract config {config_path}: {exc}") from exc
    value = config.get("instance_id") if isinstance(config, dict) else None
    return value if isinstance(value, str) and value else path.name


def contract_environment_id(contract_dir: str | Path) -> int:
    """Read the canonical environment ID from a contract's config."""
    path = Path(contract_dir) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load contract config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    value = next(
        (
            config[key]
            for key in ("evaluation_environment_id", "environment_id")
            if config.get(key) is not None
        ),
        None,
    )
    if value is None:
        metadata = config.get("synthetic_contract_metadata") or {}
        if isinstance(metadata, dict):
            value = next(
                (
                    metadata[key]
                    for key in (
                        "evaluation_environment_id",
                        "environment_id",
                    )
                    if metadata.get(key) is not None
                ),
                None,
            )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Contract config must declare an integer environment_id: {path}"
        )
    if value not in range(1, 7):
        raise ValueError(f"Contract environment_id must be between 1 and 6: {path}")
    return value


def validate_contract_environment(
    contract_dir: str | Path,
    environment_id: int,
) -> None:
    """Reject a contract prepared for a different environment."""
    actual = contract_environment_id(contract_dir)
    if actual != environment_id:
        raise ValueError(
            f"Contract {Path(contract_dir).name!r} targets environment {actual}, "
            f"not requested environment {environment_id}"
        )


__all__ = (
    "contract_environment_id",
    "contract_id",
    "validate_contract_environment",
)
