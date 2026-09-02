"""Thin indexing for the canonical algorithm contract corpus."""

from __future__ import annotations

import json
from pathlib import Path

from performance.contracts import contract_id, validate_contract_environment
from performance.orchestration.run import EXCLUDED_BATCH_VARIANTS


def index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Contracts root does not exist or is not a directory: {root}")
    contracts: dict[str, Path] = {}
    for text_path in sorted(root.rglob("contract.txt")):
        directory = text_path.parent
        if any(part.startswith(".") for part in directory.relative_to(root).parts):
            continue
        if not (directory / "config.json").is_file() or not (directory / "terms.json").is_file():
            continue
        stable_id = contract_id(directory)
        if stable_id in contracts:
            raise ValueError(f"Duplicate contract ID {stable_id!r} under {root}")
        contracts[stable_id] = directory
    return contracts


def select(root: Path, requested: list[str] | None, prefix: str | None, limit: int | None) -> list[str]:
    discovered = index(root)
    if requested:
        missing = sorted(set(requested) - set(discovered))
        if missing:
            raise FileNotFoundError("Unknown contract IDs: " + ", ".join(missing))
        selected = list(requested)
    else:
        selected = sorted(discovered)
    if prefix:
        selected = [item for item in selected if item.startswith(prefix)]
    selected = [
        item for item in selected
        if json.loads(
            (discovered[item] / "config.json").read_text(encoding="utf-8")
        ).get("contingency_variant") not in EXCLUDED_BATCH_VARIANTS
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise FileNotFoundError(f"No contracts selected under {root}")
    for stable_id in selected:
        validate_contract_environment(discovered[stable_id], 5)
    return selected


def load(root: Path, stable_id: str) -> dict:
    directory = index(root).get(stable_id)
    if directory is None:
        raise FileNotFoundError(f"Unknown contract ID {stable_id!r} under {root}")
    validate_contract_environment(directory, 5)

    def read(name: str) -> dict:
        return json.loads((directory / name).read_text(encoding="utf-8"))

    metadata_path = directory / "metadata.json"
    return {
        "contract_dir": directory,
        "config": read("config.json"),
        "terms": read("terms.json"),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {},
    }
