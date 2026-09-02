"""Load the preserved domain prompts with shared rendering primitives."""

from __future__ import annotations

import re
from pathlib import Path

from domains import DOMAINS
from llm_utils import (
    CONTRACT_BEGIN,
    CONTRACT_END,
    delimit_contract,
)


ROOT = Path(__file__).resolve().parents[1] / "domain-prompts"
CONTRACT_TOKEN = "{{NEGOTIATED_CONTRACT}}"
_FORBIDDEN = {
    "hotel-cleaning": (
        "caterer", "catering", "customer", "dish", "ingredient", "margherita",
        "mozzarella", "pasta", "pizza", "production", "soup",
    ),
    "ai-model-hosting": (
        "caterer", "catering", "cleaning", "customer", "detergent", "dish",
        "disinfectant", "hotel", "ingredient", "margherita", "mozzarella",
        "pasta", "pizza", "production", "soup", "wet-wipes",
    ),
}
_REQUIRED = {
    "hotel-cleaning": (
        "hotel owner", "cleaning provider", "Conference Room Cleaning",
        "King-Sized Room Cleaning", "Single Room Cleaning", "Detergent",
        "Disinfectant", "Wet-wipes", "$20,000", "$2,000", "multiples of $100",
    ),
    "ai-model-hosting": (
        "AI startup founder", "AI service provider", "Fine-tuning", "Model-hosting",
        "Inference", "CPU", "Memory", "GPU", "$200,000", "$20,000",
        "multiples of $1,000",
    ),
}
_HEADERS = {
    "hotel-cleaning": (
        "CLEANING WEEK PROCESS (in order):", "CLEANING-SERVICE REQUIREMENTS:",
        "CLEANING-SERVICE SUBSTITUTABILITY:", "CONTRACT EXECUTION ACTIONS:",
        "ENVIRONMENT STOCHASTICITY:", "MINIMUM REQUIREMENTS:",
        "PRIVATE VALUATION:", "OBJECTIVES:", "SCHEDULE:",
    ),
    "ai-model-hosting": (
        "SERVICE WEEK PROCESS (in order):", "SERVICE REQUIREMENTS:",
        "SERVICE SUBSTITUTABILITY:", "CONTRACT EXECUTION ACTIONS:",
        "ENVIRONMENT STOCHASTICITY:", "MINIMUM REQUIREMENTS:",
        "PRIVATE VALUATION:", "OBJECTIVES:", "SCHEDULE:",
    ),
}


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Fixed prompt is empty: {path}")
    return text


def _leaks(domain: str, text: str) -> list[str]:
    lowered = text.casefold()
    return [
        term for term in _FORBIDDEN[domain]
        if re.search(rf"\b{re.escape(term)}(?:es|s)?\b", lowered)
    ]


def validate(domain: str) -> None:
    _validate_fixed(domain)


def _validate_fixed(domain: str) -> None:
    root = ROOT / domain / "performance"
    combined = []
    for role in ("customer", "supplier"):
        info = _read(root / f"{role}_info.txt")
        goal = _read(root / f"{role}_goal.txt")
        if info.count(CONTRACT_TOKEN) != 1:
            raise RuntimeError(f"{role}_info.txt must contain {CONTRACT_TOKEN} exactly once")
        if CONTRACT_TOKEN in goal:
            raise RuntimeError(f"{role}_goal.txt must not contain {CONTRACT_TOKEN}")
        combined.extend((info.replace(CONTRACT_TOKEN, ""), goal))
    text = "\n".join(combined)
    if "{{" in text or "}}" in text:
        raise RuntimeError(f"Fixed {domain} prompts contain unresolved variables")
    leaked = _leaks(domain, text)
    if leaked:
        raise RuntimeError(f"Fixed {domain} prompts contain source-domain terms: " + ", ".join(leaked))
    missing = [phrase for phrase in _REQUIRED[domain] if phrase not in text]
    if missing:
        raise RuntimeError(f"Fixed {domain} prompts are missing required content: " + ", ".join(missing))


def build(
    domain: str,
    contract: str,
    treatment: str = "control",
    treated_role: str | None = None,
) -> list[dict[str, str]]:
    del treated_role
    contract = str(contract).strip()
    if not contract:
        raise RuntimeError("Performance prompts require a non-empty contract")
    root = ROOT / domain / "performance"
    if treatment != "control":
        raise ValueError("fixed domain prompts support the control treatment only")
    _validate_fixed(domain)
    leaked = _leaks(domain, contract)
    if leaked:
        raise RuntimeError(f"Model-facing {domain} contract contains source-domain terms: " + ", ".join(leaked))
    names = DOMAINS[domain]["agent_names"]
    players = []
    for name, prompt_role, canonical_role in zip(
        names,
        ("customer", "supplier"),
        ("customer", "supplier"),
        strict=True,
    ):
        info = _read(root / f"{prompt_role}_info.txt").replace(
            CONTRACT_TOKEN,
            delimit_contract(contract),
        )
        player = {
            "name": name,
            "canonical_role": canonical_role,
            "info": info,
            "goal": _read(root / f"{prompt_role}_goal.txt"),
        }
        if info.count(CONTRACT_BEGIN) != 1 or info.count(CONTRACT_END) != 1:
            raise RuntimeError(f"{name} prompt must contain exactly one delimited contract")
        players.append(player)
    return players


def formatter(domain: str):
    """Return the established readability formatter for one domain."""
    headers = _HEADERS[domain]
    pattern = re.compile(
        r"(^| )(" + "|".join(re.escape(header) for header in headers) + r") ?",
        flags=re.MULTILINE,
    )

    def format_text(text: str) -> str:
        def format_chunk(chunk: str) -> str:
            return pattern.sub(
                lambda match: ("" if match.group(1) == "" else "\n\n") + match.group(2) + "\n",
                chunk,
            )

        pieces = str(text).split(CONTRACT_BEGIN)
        formatted = [format_chunk(pieces[0])]
        for piece in pieces[1:]:
            if CONTRACT_END not in piece:
                formatted.extend((CONTRACT_BEGIN, piece))
                continue
            contract, remainder = piece.split(CONTRACT_END, 1)
            formatted.extend((CONTRACT_BEGIN, contract, CONTRACT_END, format_chunk(remainder)))
        return "".join(formatted).strip()

    def format_system_prompt(goal: str, info: str) -> str:
        return f"Goal:\n\n{format_text(goal)}\n\nPrivate Info:\n\n{format_text(info)}"

    return format_system_prompt
