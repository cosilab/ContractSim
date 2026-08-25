"""Render customer and supplier prompts for contract performance."""

from __future__ import annotations

from pathlib import Path

from llm_utils import (
    DEFAULT_ENVIRONMENT_ROOT,
    EnvironmentLoadError,
    PromptRenderError,
    delimit_contract,
    load_prompt_context,
    render_template,
)

from .ablations import CONTROL_TREATMENT, append_goal_treatment


PERFORMANCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_PROMPT_ROOT = PERFORMANCE_ROOT / "domain-prompts" / "catering"
ROLES = ("customer", "supplier")
PROMPT_COMPONENTS = ("info", "goal")


class PromptBuildError(ValueError):
    """Raised when performance prompts cannot be rendered."""


def build_prompts(
    environment_id: int,
    negotiated_contract: str,
    domain_prompt_root: str | Path = DEFAULT_DOMAIN_PROMPT_ROOT,
    goal_treatment: str = CONTROL_TREATMENT,
    treated_role: str | None = None,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> list[dict[str, str]]:
    """Build private customer and supplier prompts for one contract."""
    if not str(negotiated_contract).strip():
        raise PromptBuildError("negotiated_contract must be non-empty")

    try:
        context = load_prompt_context(environment_id, environment_root)
    except EnvironmentLoadError as exc:
        raise PromptBuildError(str(exc)) from exc

    variables = {
        str(key): value
        for key, value in context["template_variables"].items()
    }
    variables["NEGOTIATED_CONTRACT"] = delimit_contract(negotiated_contract)
    prompt_root = Path(domain_prompt_root)

    players: list[dict[str, str]] = []
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

    try:
        return append_goal_treatment(players, goal_treatment, treated_role)
    except ValueError as exc:
        raise PromptBuildError(str(exc)) from exc
