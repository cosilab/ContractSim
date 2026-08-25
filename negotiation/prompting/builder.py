"""Render customer and caterer prompts for a resolved environment."""

from __future__ import annotations

from pathlib import Path

from llm_utils import (
    DEFAULT_ENVIRONMENT_ROOT,
    EnvironmentLoadError,
    PromptRenderError,
    load_prompt_context,
    render_template,
)

from .ablations import CONTROL_TREATMENT, append_goal_treatment


NEGOTIATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_PROMPT_ROOT = NEGOTIATION_ROOT / "domain-prompts" / "catering"

ROLES = ("customer", "caterer")
PROMPT_COMPONENTS = ("info", "goal")


class PromptBuildError(ValueError):
    """Raised when environment data or a domain template cannot be rendered."""


def _validate_max_rounds(max_rounds: int) -> None:
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 1
    ):
        raise PromptBuildError("max_rounds must be a positive integer")


def build_prompts(
    environment_id: int,
    domain_prompt_root: str | Path = DEFAULT_DOMAIN_PROMPT_ROOT,
    goal_treatment: str = CONTROL_TREATMENT,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
    max_rounds: int = 50,
) -> list[dict[str, str]]:
    """Build customer and caterer negotiation prompts for one environment.

    ``domain_prompt_root`` must contain ``customer`` and ``caterer`` directories
    with ``info.txt`` and ``goal.txt`` templates.
    """
    _validate_max_rounds(max_rounds)

    try:
        context = load_prompt_context(environment_id, environment_root)
    except EnvironmentLoadError as exc:
        raise PromptBuildError(str(exc)) from exc
    variables = {
        str(key): value
        for key, value in context["template_variables"].items()
    }
    variables["MAX_ROUNDS"] = max_rounds
    prompt_root = Path(domain_prompt_root)

    players = []
    for role in ROLES:
        player = {"name": role}
        for component in PROMPT_COMPONENTS:
            try:
                player[component] = render_template(
                    prompt_root / role / f"{component}.txt",
                    variables,
                )
            except PromptRenderError as exc:
                raise PromptBuildError(str(exc)) from exc
        players.append(player)
    return append_goal_treatment(players, goal_treatment)
