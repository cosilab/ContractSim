"""Named prompt-level goal ablations for negotiation experiments."""

from __future__ import annotations

CONTROL_TREATMENT = "control"
ROLES = ("customer", "caterer")


TREATMENTS = {
    "contingencies": {
        "customer": (
            "RISK: You should be wary of the risk of default from the "
            "Caterer. Consider adding contingency clauses."
        ),
        "caterer": (
            "RISK: Be wary of the risk of default from the Customer and "
            "environment stochasticity. Consider adding contingency clauses."
        ),
    },
    "net-utility": {
        role: (
            "ACCEPTING CONTRACTS: You should only accept contracts if you "
            "expect them to result in a net increase in utility."
        )
        for role in ROLES
    },
    "no-worse-prior": {
        role: (
            "ACCEPTING CONTRACTS: You should not accept contracts that are "
            "worse for you than any contract that has been proposed to you "
            "before."
        )
        for role in ROLES
    },
}

ABLATION_ORDER = tuple(TREATMENTS)
GOAL_TREATMENT_CHOICES = (CONTROL_TREATMENT, *ABLATION_ORDER)


def get_treatment(name: str) -> dict[str, str] | None:
    """Resolve a named treatment; control deliberately appends nothing."""
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
) -> list[dict[str, str]]:
    """Return copied player configs with one exact clause appended to each goal."""
    treatment = get_treatment(treatment_name)
    rendered: list[dict[str, str]] = []
    for player in players:
        copied = dict(player)
        role = str(copied["name"]).lower()
        if role not in ROLES:
            raise ValueError(f"Unknown negotiation role {role!r}")
        if treatment is not None:
            copied["goal"] = (
                f"{str(copied['goal']).rstrip()}\n\n{treatment[role]}"
            )
        rendered.append(copied)
    return rendered


def treatment_metadata(name: str) -> dict[str, object]:
    """Describe the selected prompt ablation in the existing artifact schema."""
    treatment = get_treatment(name)
    return {
        "goal_treatment": name,
        "goal_clauses": dict(treatment) if treatment is not None else {},
        "goal_clause_location": "goal_component_append",
    }
