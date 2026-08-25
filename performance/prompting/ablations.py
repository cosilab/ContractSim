"""Named goal ablations for performance experiments."""

from __future__ import annotations

from dataclasses import dataclass


CONTROL_TREATMENT = "control"
ROLES = ("customer", "supplier")


@dataclass(frozen=True)
class GoalTreatment:
    clause: str
    counterparty: str


TREATMENTS = {
    "plan-ahead": GoalTreatment(
        clause=(
            "CONTRACT PERFORMANCE: Plan ahead under uncertainty, and consider "
            "multiple possible outcomes."
        ),
        counterparty="rcc",
    ),
    "no-uni": GoalTreatment(
        clause=(
            "CONTRACT PERFORMANCE: Avoid unilaterally violating the contract, "
            "and plan ahead so that you aren't in a position when you are forced "
            "to violate the terms."
        ),
        counterparty="rcc",
    ),
    "institutional": GoalTreatment(
        clause=(
            "CONTRACT PERFORMANCE: You are in a repeated game; violating the "
            "contract may result in you losing out on future opportunities to "
            "make contracts with the same counter-party in the future."
        ),
        counterparty="rcc",
    ),
    "recip-def": GoalTreatment(
        clause=(
            "CONTRACT PERFORMANCE: Stop following the contract once it's clear "
            "that the other party is not following the contract."
        ),
        counterparty="re",
    ),
}

GOAL_TREATMENT_CHOICES = (CONTROL_TREATMENT, *TREATMENTS)


def get_treatment(name: str) -> GoalTreatment | None:
    """Resolve a treatment; control appends no clause."""
    if name == CONTROL_TREATMENT:
        return None
    try:
        return TREATMENTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown goal treatment {name!r}; expected one of "
            + ", ".join(GOAL_TREATMENT_CHOICES)
        ) from exc


def append_goal_treatment(
    players: list[dict[str, str]],
    treatment_name: str,
    treated_role: str | None,
) -> list[dict[str, str]]:
    """Append a named clause only to the focal performance role."""
    treatment = get_treatment(treatment_name)
    if treated_role is not None and treated_role not in ROLES:
        raise ValueError(f"Unknown performance role {treated_role!r}")
    if treatment is not None and treated_role is None:
        raise ValueError("treated_role is required for a named goal treatment")

    rendered: list[dict[str, str]] = []
    for player in players:
        copied = dict(player)
        if treatment is not None and copied["name"] == treated_role:
            copied["goal"] = (
                f"{copied['goal'].rstrip()}\n\n{treatment.clause}"
            )
        rendered.append(copied)
    return rendered


def treatment_metadata(
    name: str,
    treated_role: str | None = None,
) -> dict[str, str | None]:
    """Describe the selected performance prompt treatment."""
    treatment = get_treatment(name)
    return {
        "goal_treatment": name,
        "goal_treatment_role": treated_role,
        "goal_clause": treatment.clause if treatment else None,
        "counterparty_baseline": treatment.counterparty if treatment else None,
    }
