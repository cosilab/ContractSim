"""Phase-neutral building blocks shared across RatConBench."""

from .environments import (
    DEFAULT_ENVIRONMENT_ROOT,
    EnvironmentLoadError,
    load_environment,
    load_environment_metadata,
    load_prompt_context,
)
from .prompting import (
    CONTRACT_BEGIN,
    CONTRACT_END,
    PromptRenderError,
    delimit_contract,
    format_system_prompt,
    render_template,
)

__all__ = (
    "CONTRACT_BEGIN",
    "CONTRACT_END",
    "DEFAULT_ENVIRONMENT_ROOT",
    "EnvironmentLoadError",
    "PromptRenderError",
    "delimit_contract",
    "format_system_prompt",
    "load_environment",
    "load_environment_metadata",
    "load_prompt_context",
    "render_template",
)
