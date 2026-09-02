"""The two model-facing views layered over canonical Env 5 contracts."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from domains import DOMAINS


_MONEY_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)")
_VIEWS = {
    "hotel-cleaning": {
        "parties": ("Hotel Owner", "Cleaning Provider"),
        "forbidden": re.compile(
            r"\b(?:customer|caterer|catering|dish(?:es)?|margherita|pizza(?:s)?|"
            r"pasta(?:s)?|soup(?:s)?|production)\b", re.IGNORECASE
        ),
        "replacements": (
            ("Parties: Customer and Caterer.", "Parties: Hotel Owner and Cleaning Provider."),
            ("Margherita Pizzas", "Conference Room Cleanings"),
            ("Margherita Pizza", "Conference Room Cleaning"),
            ("Pesto Pastas", "King-Sized Room Cleanings"),
            ("Pesto Pasta", "King-Sized Room Cleaning"),
            ("Tomato Soups", "Single Room Cleanings"),
            ("Tomato Soup", "Single Room Cleaning"),
            ("Delivery obligations:", "Service obligations:"),
            ("delivery obligation", "service obligation"),
            ("actual physical deliveries", "actual provided services"),
            ("After delivering", "After performing"),
            ("production week's delivery", "cleaning week's service quantity"),
            ("production round's shortfall", "cleaning round's shortfall"),
            ("production weeks", "cleaning weeks"),
            ("production week", "cleaning week"),
            ("production round", "cleaning round"),
            ("Delivery:", "Cleaning services:"),
            ("undelivered units", "unperformed cleaning services"),
            ("delivers", "performs"),
            ("delivered", "performed"),
            ("dishes", "cleaning types"),
            ("dish", "cleaning type"),
            ("Customer", "Hotel Owner"),
            ("customer", "hotel owner"),
            ("Caterer", "Cleaning Provider"),
            ("caterer", "cleaning provider"),
        ),
    },
    "ai-model-hosting": {
        "parties": ("AI Startup Founder", "AI Service Provider"),
        "forbidden": re.compile(
            r"\b(?:customer|caterer|catering|dish(?:es)?|ingredient(?:s)?|"
            r"margherita|mozzarella|pasta(?:s)?|pizza(?:s)?|production|soup(?:s)?|"
            r"hotel|cleaning|detergent|disinfectant)\b|wet-wipes?", re.IGNORECASE
        ),
        "replacements": (
            ("Parties: Customer and Caterer.", "Parties: AI Startup Founder and AI Service Provider."),
            ("Weekly minimums:", "Service-round minimums:"),
            ("weekly minimums", "service-round minimums"),
            ("Margherita Pizzas", "Fine-tuning services"),
            ("Margherita Pizza", "Fine-tuning service"),
            ("Pesto Pastas", "Model-hosting services"),
            ("Pesto Pasta", "Model-hosting service"),
            ("Tomato Soups", "Inference services"),
            ("Tomato Soup", "Inference service"),
            ("Delivery obligations:", "Service obligations:"),
            ("delivery obligation", "service obligation"),
            ("actual physical deliveries", "actual provided services"),
            ("After delivering", "After providing"),
            ("production week's delivery", "service week's provision"),
            ("production round's shortfall", "service round's shortfall"),
            ("production weeks", "service weeks"),
            ("production week", "service week"),
            ("production round", "service round"),
            ("Delivery:", "Service provision:"),
            ("undelivered units", "unprovided services"),
            ("delivers", "provides"),
            ("delivered", "provided"),
            ("dishes", "services"),
            ("dish", "service"),
            ("Customer", "AI Startup Founder"),
            ("customer", "AI startup founder"),
            ("Caterer", "AI Service Provider"),
            ("caterer", "AI service provider"),
        ),
    },
}


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid monetary value: {value!r}") from exc


def display_amount(domain: str, canonical_value: Any) -> int | float:
    scaled = _decimal(canonical_value) * DOMAINS[domain]["money_scale"]
    return int(scaled) if scaled == scaled.to_integral_value() else float(scaled)


def format_money(domain: str, canonical_value: Any) -> str:
    scaled = _decimal(canonical_value) * DOMAINS[domain]["money_scale"]
    return f"${int(scaled):,}" if scaled == scaled.to_integral_value() else f"${scaled:,.2f}"


def _adapt_contract(domain: str, contract: str) -> str:
    source = str(contract).strip()
    if not source:
        raise ValueError("Cannot adapt an empty contract")
    if "Customer" not in source or "Caterer" not in source:
        raise ValueError("Expected a canonical source contract with both parties")
    view = _VIEWS[domain]
    adapted = source
    for old, new in view["replacements"]:
        adapted = adapted.replace(old, new)
    scale = DOMAINS[domain]["money_scale"]

    def scale_money(match: re.Match[str]) -> str:
        value = _decimal(match.group(1)) * scale
        return f"${int(value):,}" if value == value.to_integral_value() else f"${format(value.normalize(), 'f')}"

    adapted = _MONEY_RE.sub(scale_money, adapted)
    leak = view["forbidden"].search(adapted)
    if leak:
        raise ValueError(f"{domain} contract adaptation left source term {leak.group(0)!r}")
    for raw in _MONEY_RE.findall(adapted):
        if _decimal(raw) % scale != 0:
            raise ValueError(f"Adapted contract amount ${raw} is not a multiple of ${scale:,}")
    return adapted


def load_contract_views(contract_dir: Path, domain: str) -> tuple[str, str, bool]:
    """Return canonical and model-facing text without copying any contracts."""
    stored = (contract_dir / "contract.txt").read_text(encoding="utf-8").strip()
    canonical_path = contract_dir / "canonical_contract.txt"
    if not canonical_path.is_file():
        return stored, _adapt_contract(domain, stored), False

    canonical = canonical_path.read_text(encoding="utf-8").strip()
    first, second = _VIEWS[domain]["parties"]
    if "Customer" not in canonical or "Caterer" not in canonical:
        raise ValueError(f"Invalid canonical contract: {canonical_path}")
    if first not in stored or second not in stored:
        raise ValueError(f"Invalid materialized {domain} contract: {contract_dir / 'contract.txt'}")
    leak = _VIEWS[domain]["forbidden"].search(stored)
    if leak:
        raise ValueError(f"Materialized {domain} contract contains source term {leak.group(0)!r}")
    scale = DOMAINS[domain]["money_scale"]
    for raw in _MONEY_RE.findall(stored):
        if _decimal(raw) % scale != 0:
            raise ValueError(f"Materialized {domain} contract amount ${raw} is not a multiple of ${scale:,}")
    return canonical, stored, True


def metadata(domain: str) -> dict[str, Any]:
    spec = DOMAINS[domain]
    customer, supplier = spec["agent_names"]
    result: dict[str, Any] = {
        "domain": domain,
        "environment_id": 5,
        "source_environment_id": 5,
        "source_environment_name": "env-05_baseline_current_cap20",
        "monetary_scale": spec["money_scale"],
        "model_facing_money": "displayed_usd",
        "analysis_money": "canonical_env5_units",
        "canonical_money_rule": f"displayed_money / {spec['money_scale']}",
        "role_mapping": {customer: "customer", supplier: "supplier"},
        "service_mapping": {
            spec["services"]["A"]: "Margherita Pizza",
            spec["services"]["B"]: "Pesto Pasta",
            spec["services"]["C"]: "Tomato Soup",
        },
    }
    material_key = "supply_mapping" if domain == "hotel-cleaning" else "resource_mapping"
    result[material_key] = {
        spec["materials"]["Z"]: "Tomato",
        spec["materials"]["X"]: "Mozzarella",
        spec["materials"]["Y"]: "Basil",
    }
    return result
