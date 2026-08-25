"""Build performance-phase prompts from environment data and templates."""

from .ablations import (
    CONTROL_TREATMENT,
    GOAL_TREATMENT_CHOICES,
    TREATMENTS,
    treatment_metadata,
)
from .builder import DEFAULT_DOMAIN_PROMPT_ROOT, PromptBuildError, build_prompts

__all__ = (
    "CONTROL_TREATMENT",
    "DEFAULT_DOMAIN_PROMPT_ROOT",
    "GOAL_TREATMENT_CHOICES",
    "PromptBuildError",
    "TREATMENTS",
    "build_prompts",
    "treatment_metadata",
)
