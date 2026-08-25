"""Load canonical resolved environment documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT_ROOT = PROJECT_ROOT / "envs"


class EnvironmentLoadError(ValueError):
    """Raised when a resolved environment document cannot be loaded safely."""


def _environment_file(
    environment_id: int,
    filename: str,
    environment_root: str | Path,
) -> Path:
    if isinstance(environment_id, bool) or not isinstance(environment_id, int):
        raise EnvironmentLoadError("environment_id must be an integer")
    if environment_id < 1:
        raise EnvironmentLoadError("environment_id must be positive")
    return Path(environment_root) / str(environment_id) / filename


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EnvironmentLoadError(f"Missing environment input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EnvironmentLoadError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise EnvironmentLoadError(f"Cannot read environment input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvironmentLoadError(f"Expected a JSON object in {path}")
    return value


def _load_environment_document(
    environment_id: int,
    filename: str,
    environment_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    path = _environment_file(environment_id, filename, environment_root)
    document = _load_json_object(path)
    declared_id = document.get("environment_id")
    if (
        isinstance(declared_id, bool)
        or not isinstance(declared_id, int)
        or declared_id != environment_id
    ):
        raise EnvironmentLoadError(
            f"Environment ID in {path} does not match directory {environment_id}"
        )
    return path, document


def load_environment(
    environment_id: int,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> dict[str, Any]:
    """Load the complete executable snapshot for one environment."""
    _path, environment = _load_environment_document(
        environment_id,
        "environment.json",
        environment_root,
    )
    return environment


def load_prompt_context(
    environment_id: int,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> dict[str, Any]:
    """Load prompt-safe template substitutions for one environment."""
    path, context = _load_environment_document(
        environment_id,
        "prompt-context.json",
        environment_root,
    )
    variables = context.get("template_variables")
    if not isinstance(variables, dict):
        raise EnvironmentLoadError(f"Missing template_variables object in {path}")
    if not all(isinstance(key, str) for key in variables):
        raise EnvironmentLoadError(
            f"template_variables keys must be strings in {path}"
        )
    return context


def load_environment_metadata(
    environment_id: int,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> dict[str, object]:
    """Return the stable identity and experimental axes for one environment."""
    path, environment = _load_environment_document(
        environment_id,
        "environment.json",
        environment_root,
    )
    name = environment.get("environment_name")
    axes = environment.get("axes")
    if not isinstance(name, str) or not name.strip():
        raise EnvironmentLoadError(f"Invalid environment_name in {path}")
    if not isinstance(axes, dict):
        raise EnvironmentLoadError(f"Invalid environment axes in {path}")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in axes.items()):
        raise EnvironmentLoadError(f"Environment axes must map strings to strings in {path}")
    return {
        "environment_id": environment_id,
        "environment_name": name,
        "environment_axes": dict(axes),
    }
