"""Explicit, phase-neutral provider and LLM runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYS_DIR = PROJECT_ROOT / "keys" / "cosi"
PROVIDERS = ("gemini", "openai", "anthropic")

DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-opus-5",
}
DEFAULT_REASONING = {
    "gemini": "high",
    "openai": "high",
    "anthropic": "high",
}
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"

_MODEL_ENVIRONMENT_NAMES = {
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}
_REASONING_ENVIRONMENT_NAMES = {
    "gemini": "GEMINI_THINKING_LEVEL",
    "openai": "OPENAI_REASONING_EFFORT",
    "anthropic": "ANTHROPIC_EFFORT",
}
_API_KEY_ENVIRONMENT_NAMES = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}
_API_KEY_FILES = {
    "gemini": "GEMINI_API.key",
    "openai": "OPENAI_API.key",
    "anthropic": "ANTHROPIC_API.key",
}


def _validate_provider(provider: str) -> str:
    normalized = str(provider).lower()
    if normalized not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}"
        )
    return normalized


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved model and reasoning settings for one provider."""

    provider: str
    model: str
    reasoning: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _validate_provider(self.provider))
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.reasoning.strip():
            raise ValueError("reasoning must not be empty")

    @property
    def reasoning_metadata(self) -> dict[str, str]:
        key = "thinking_level" if self.provider == "gemini" else "effort"
        return {key: self.reasoning}


@dataclass(frozen=True)
class RuntimeOptions:
    """Per-agent adapter behavior shared across providers and phases."""

    prompt_cache_enabled: bool = True
    thought_traces_enabled: bool = True
    cache_scope: str = "ratconbench"


def load_provider_config(
    provider: str,
    environ: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Resolve one provider without reading credentials or creating a client."""
    provider = _validate_provider(provider)
    values = os.environ if environ is None else environ
    model = values.get(_MODEL_ENVIRONMENT_NAMES[provider], DEFAULT_MODELS[provider])
    reasoning = values.get(
        _REASONING_ENVIRONMENT_NAMES[provider],
        DEFAULT_REASONING[provider],
    )
    return ProviderConfig(provider=provider, model=model, reasoning=reasoning)


def resolve_api_key(
    provider: str,
    *,
    keys_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read a provider key from its standard environment name or a key file."""
    provider = _validate_provider(provider)
    values = os.environ if environ is None else environ
    for variable in _API_KEY_ENVIRONMENT_NAMES[provider]:
        value = values.get(variable, "").strip()
        if value:
            return value

    configured_dir = keys_dir or values.get("KEYS_DIR") or DEFAULT_KEYS_DIR
    path = Path(configured_dir).expanduser().resolve() / _API_KEY_FILES[provider]
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        names = " or ".join(_API_KEY_ENVIRONMENT_NAMES[provider])
        raise RuntimeError(
            f"No {provider} API key found. Set {names}, or place "
            f"{_API_KEY_FILES[provider]} in {path.parent} (override with KEYS_DIR)."
        ) from exc
    if not value:
        raise RuntimeError(f"API key file is empty: {path}")
    return value
