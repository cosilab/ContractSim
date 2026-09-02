#!/usr/bin/env python3
"""Calculate descriptive and contract-quality metrics for negotiation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from contract_parser import (
    DISH_PATTERNS,
    PAYMENT_WEEKS,
    PRODUCTION_WEEKS,
    ParsedContract,
    decimal_text,
    parse_contract,
)


HERE = Path(__file__).resolve().parent
NEGOTIATION_ANALYSIS_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "negotiation-main"
INFERENCES_ROOT = NEGOTIATION_ANALYSIS_ROOT / "inferences"
DEFAULT_ENV_ROOT = REPO_ROOT / "envs"


def campaign_runs_root(campaign: str) -> Path:
    return RESULTS_ROOT / campaign


def campaign_inferences_root(campaign: str) -> Path:
    return INFERENCES_ROOT / campaign


@dataclass(frozen=True)
class EnvironmentFacts:
    environment_id: int
    environment_name: str
    operating_conditions: str
    customer_preference: str
    caterer_capital: str
    customer_budget: Decimal
    customer_outside_option: Decimal
    customer_values: dict[str, Decimal]
    minimum_quantities: dict[str, int]
    recipes: dict[str, dict[str, int]]
    inventory_cap: int
    caterer_starting_balance: Decimal


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment_facts(environment_id: int, env_root: Path) -> EnvironmentFacts:
    payload = read_json(env_root / str(environment_id) / "environment.json")
    base = payload["base_game"]
    customer = base["customer"]
    return EnvironmentFacts(
        environment_id=environment_id,
        environment_name=payload["environment_name"],
        operating_conditions=payload["axes"]["operating_conditions"],
        customer_preference=payload["axes"]["customer_preference"],
        caterer_capital=payload["axes"]["caterer_capital"],
        customer_budget=Decimal(str(customer["starting_budget"])),
        customer_outside_option=Decimal(str(customer["outside_option_utility"])),
        customer_values={
            dish: Decimal(str(value))
            for dish, value in payload["customer_preference"]["private_values"].items()
        },
        minimum_quantities={
            dish: int(value)
            for dish, value in customer["minimum_dishes_per_production_round"].items()
        },
        recipes={
            dish: {ingredient: int(value) for ingredient, value in recipe.items()}
            for dish, recipe in base["recipes"].items()
        },
        inventory_cap=int(base["inventory"]["max_units_per_ingredient"]),
        caterer_starting_balance=Decimal(str(payload["caterer_capital"]["starting_balance"])),
    )


def discover_matchups(runs_root: Path) -> list[Path]:
    return sorted(
        set(runs_root.glob("env-*/run */*_vs_*"))
        | set(runs_root.glob("env-*/run-*/*_vs_*"))
    )


def _canonical_environment_id(matchup_dir: Path) -> int:
    match = re.fullmatch(r"env-(\d+)", matchup_dir.parents[1].name)
    if match is None:
        raise ValueError(
            f"Unexpected environment directory: {matchup_dir.parents[1]}"
        )
    return int(match.group(1))


def _run_number(path: Path) -> int:
    match = re.fullmatch(r"run[ -](\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Unexpected run directory: {path.parent}")
    return int(match.group(1))


def _termination(log_text: str, summary: dict[str, Any]) -> str:
    if not summary.get("valid", True):
        return "invalid"
    if summary.get("accepted"):
        return "agreement"
    if ">>> Customer terminated" in log_text:
        return "customer_terminated"
    if ">>> Caterer terminated" in log_text:
        return "caterer_terminated"
    if ">>> Max rounds reached" in log_text:
        return "max_rounds"
    return "no_agreement_other"


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_offers(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(trace_rows, start=1):
        parsed = trace.get("parsed_json") or {}
        contract = str(parsed.get("contract") or "").strip()
        if not contract:
            continue
        offers.append(
            {
                "offer_index": len(offers) + 1,
                "trace_index": trace_index,
                "round": int(trace.get("round") or 0),
                "proposer_role": str(trace.get("agent") or ""),
                "proposer_provider": str(trace.get("provider") or ""),
                "proposer_model": str(trace.get("model") or ""),
                "contract_text": contract,
            }
        )
    return offers


def extract_offers_from_log(
    log_text: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recover formal proposals when optional reasoning traces were disabled."""
    lines = log_text.splitlines()
    offers: list[dict[str, Any]] = []
    round_number = 0
    index = 0
    while index < len(lines):
        round_match = re.fullmatch(r"--- Round (\d+) of \d+ ---", lines[index].strip())
        if round_match is not None:
            round_number = int(round_match.group(1))
            index += 1
            continue

        contract_match = re.fullmatch(
            r"\[CONTRACT by (customer|caterer)\]:",
            lines[index].strip(),
        )
        if contract_match is None:
            index += 1
            continue

        role = contract_match.group(1)
        index += 1
        contract_lines: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if (
                re.fullmatch(r"--- Round \d+ of \d+ ---", stripped)
                or re.match(r"^(customer|caterer):", stripped)
                or stripped.startswith(">>>")
                or stripped == "[FINAL CONTRACT]:"
                or re.fullmatch(r"\[CONTRACT by (customer|caterer)\]:", stripped)
            ):
                break
            contract_lines.append(lines[index])
            index += 1

        contract = "\n".join(contract_lines).strip()
        if contract:
            offers.append(
                {
                    "offer_index": len(offers) + 1,
                    "trace_index": None,
                    "round": round_number,
                    "proposer_role": role,
                    "proposer_provider": str(summary.get(f"{role}_provider") or ""),
                    "proposer_model": str(summary.get(f"{role}_model") or ""),
                    "contract_text": contract,
                }
            )
    return offers


def _resource_issues(parsed: ParsedContract, facts: EnvironmentFacts) -> list[str]:
    issues: list[str] = []
    ingredients = sorted(
        {ingredient for recipe in facts.recipes.values() for ingredient in recipe}
    )
    for week, quantities in parsed.deliveries.items():
        for ingredient in ingredients:
            required = sum(
                quantities[dish] * facts.recipes[dish].get(ingredient, 0)
                for dish in quantities
            )
            if required > facts.inventory_cap:
                issues.append(f"inventory_cap:week={week}:ingredient={ingredient}:required={required}")
    return issues


def contract_issues(parsed: ParsedContract, facts: EnvironmentFacts) -> dict[str, list[str]]:
    structural = list(parsed.parse_issues)
    payment_total = parsed.scheduled_payment_total
    unit_total = parsed.unit_price_total
    if payment_total is not None and unit_total is not None and payment_total != unit_total:
        structural.append(
            f"unit_price_payment_mismatch:unit_price_total={decimal_text(unit_total)}:"
            f"payment_total={decimal_text(payment_total)}"
        )
    if payment_total is not None:
        for reported in parsed.reported_payment_totals:
            if reported != payment_total:
                structural.append(
                    f"reported_payment_mismatch:reported={decimal_text(reported)}:"
                    f"payment_total={decimal_text(payment_total)}"
                )
    if unit_total is not None:
        for reported in parsed.reported_contract_values:
            if reported != unit_total:
                structural.append(
                    f"reported_contract_value_mismatch:reported={decimal_text(reported)}:"
                    f"unit_price_total={decimal_text(unit_total)}"
                )

    environment: list[str] = []
    if payment_total is not None and payment_total > facts.customer_budget:
        environment.append(
            f"customer_budget:payment={decimal_text(payment_total)}:"
            f"budget={decimal_text(facts.customer_budget)}"
        )
    for week, quantities in parsed.deliveries.items():
        for dish, minimum in facts.minimum_quantities.items():
            if quantities[dish] < minimum:
                environment.append(
                    f"minimum_quantity:week={week}:dish={dish}:"
                    f"quantity={quantities[dish]}:minimum={minimum}"
                )
    environment.extend(_resource_issues(parsed, facts))
    return {"structural": structural, "environment": environment}


def _nominal_metrics(parsed: ParsedContract, facts: EnvironmentFacts) -> dict[str, Any]:
    quantities = parsed.quantity_totals
    payment_total = parsed.scheduled_payment_total
    if quantities is None or payment_total is None:
        return {}
    customer_value = sum(
        facts.customer_values[dish] * quantities[dish] for dish in quantities
    )
    customer_utility = facts.customer_budget - payment_total + customer_value
    soup_equivalents = 4 * quantities["A"] + 2 * quantities["B"] + quantities["C"]
    weighted_payment_week = None
    if payment_total > 0:
        weighted_payment_week = sum(
            Decimal(week) * amount for week, amount in parsed.payments.items()
        ) / payment_total
    return {
        "quantity_A": quantities["A"],
        "quantity_B": quantities["B"],
        "quantity_C": quantities["C"],
        "quantity_total": sum(quantities.values()),
        "quantity_soup_equivalents": soup_equivalents,
        "scheduled_payment_total": decimal_text(payment_total),
        "unit_price_total": decimal_text(parsed.unit_price_total),
        "price_payment_delta": decimal_text(
            None if parsed.unit_price_total is None else payment_total - parsed.unit_price_total
        ),
        "week1_payment": decimal_text(parsed.payments.get(1)),
        "week1_payment_share": float(parsed.payments.get(1, Decimal(0)) / payment_total)
        if payment_total > 0
        else None,
        "weighted_payment_week": float(weighted_payment_week)
        if weighted_payment_week is not None
        else None,
        "customer_nominal_value": decimal_text(customer_value),
        "customer_nominal_utility": decimal_text(customer_utility),
        "customer_nominal_surplus_vs_batna": decimal_text(
            customer_utility - facts.customer_outside_option
        ),
        "customer_nominal_individually_rational": customer_utility >= facts.customer_outside_option,
    }


def _contract_columns(parsed: ParsedContract) -> dict[str, Any]:
    columns: dict[str, Any] = {
        "price_A": decimal_text(parsed.prices["A"]),
        "price_B": decimal_text(parsed.prices["B"]),
        "price_C": decimal_text(parsed.prices["C"]),
        "contingencies": ";".join(parsed.contingencies),
        "contingency_count": len(parsed.contingencies),
    }
    for week in PRODUCTION_WEEKS:
        row = parsed.deliveries.get(week, {})
        for dish in DISH_PATTERNS:
            columns[f"delivery_w{week}_{dish}"] = row.get(dish)
    for week in PAYMENT_WEEKS:
        columns[f"payment_w{week}"] = decimal_text(parsed.payments.get(week))
    return columns


def _offer_row(
    *,
    run_key: str,
    matchup: str,
    offer: dict[str, Any],
    facts: EnvironmentFacts,
    final_text: str,
) -> dict[str, Any]:
    parsed = parse_contract(offer["contract_text"])
    issues = contract_issues(parsed, facts)
    payment_total = parsed.scheduled_payment_total
    within_budget = payment_total is not None and payment_total <= facts.customer_budget
    return {
        "run_key": run_key,
        "environment_id": facts.environment_id,
        "matchup": matchup,
        "offer_index": offer["offer_index"],
        "round": offer["round"],
        "proposer_role": offer["proposer_role"],
        "proposer_provider": offer["proposer_provider"],
        "proposer_model": offer["proposer_model"],
        "is_final": " ".join(offer["contract_text"].split()) == " ".join(final_text.split()),
        "contract_text": offer["contract_text"],
        "structurally_consistent": not issues["structural"],
        "within_budget": within_budget,
        "paper_feasible": not issues["structural"] and within_budget,
        "environment_feasible": not issues["environment"],
        "structural_issues": ";".join(issues["structural"]),
        "environment_issues": ";".join(issues["environment"]),
        **_nominal_metrics(parsed, facts),
    }


def analyze_matchup(matchup_dir: Path, facts: EnvironmentFacts) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = matchup_dir / "data" / "summary.json"
    trace_path = matchup_dir / "data" / "llm_reasoning_traces.jsonl"
    log_path = matchup_dir / "data" / "negotiation_log.txt"
    summary = read_json(summary_path)
    log_text = log_path.read_text()
    traces = _trace_rows(trace_path)
    matchup = matchup_dir.name
    run_number = _run_number(matchup_dir)
    run_key = f"env-{facts.environment_id:02d}/run-{run_number}/{matchup}"
    source_environment_id = int(summary["environment_id"])
    source_run_key = (
        f"env-{source_environment_id:02d}/run-{run_number}/{matchup}"
    )
    customer_provider, caterer_provider = matchup.split("_vs_", 1)

    parse_methods = Counter(
        str((row.get("parse") or {}).get("method") or "missing") for row in traces
    )
    if traces:
        offers = extract_offers(traces)
        offer_extraction_source = "reasoning_trace"
    else:
        offers = extract_offers_from_log(log_text, summary)
        offer_extraction_source = "negotiation_log"
    final_text = str(summary.get("contract_text") or "").strip()
    offer_rows = [
        _offer_row(
            run_key=run_key,
            matchup=matchup,
            offer=offer,
            facts=facts,
            final_text=final_text,
        )
        for offer in offers
    ]

    row: dict[str, Any] = {
        "run_key": run_key,
        "source_run_key": source_run_key,
        "environment_id": facts.environment_id,
        "source_environment_id": source_environment_id,
        "environment_name": facts.environment_name,
        "operating_conditions": facts.operating_conditions,
        "customer_preference": facts.customer_preference,
        "caterer_capital": facts.caterer_capital,
        "customer_budget": decimal_text(facts.customer_budget),
        "customer_outside_option": decimal_text(facts.customer_outside_option),
        "run_number": run_number,
        "matchup": matchup,
        "customer_provider": customer_provider,
        "customer_model": summary.get("customer_model"),
        "caterer_provider": caterer_provider,
        "caterer_model": summary.get("caterer_model"),
        "valid": bool(summary.get("valid", True)),
        "accepted": bool(summary.get("accepted")),
        "outcome": _termination(log_text, summary),
        "turns_taken": int(summary.get("turns_taken") or 0),
        "rounds_started": math.ceil(int(summary.get("turns_taken") or 0) / 2),
        "trace_count": len(traces),
        "strict_json_count": parse_methods["whole_output"],
        "recovered_json_count": parse_methods["regex_fields"] + parse_methods["last_brace_suffix"],
        "plain_text_fallback_count": parse_methods["plain_text_fallback"],
        "offer_extraction_source": offer_extraction_source,
        "offer_count": len(offers),
        "customer_offer_count": sum(offer["proposer_role"] == "customer" for offer in offers),
        "caterer_offer_count": sum(offer["proposer_role"] == "caterer" for offer in offers),
        "proposal_structural_consistency_rate": (
            sum(bool(offer["structurally_consistent"]) for offer in offer_rows) / len(offer_rows)
            if offer_rows
            else None
        ),
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "contract_path": str(matchup_dir / "data" / "contract.txt") if summary.get("accepted") else "",
    }
    if final_text:
        parsed = parse_contract(final_text)
        issues = contract_issues(parsed, facts)
        row.update(_contract_columns(parsed))
        row.update(_nominal_metrics(parsed, facts))
        row.update(
            {
                "contract_parse_complete": parsed.complete,
                "contract_structurally_consistent": not issues["structural"],
                "contract_environment_feasible": not issues["environment"],
                "contract_all_checks_pass": not issues["structural"] and not issues["environment"],
                "structural_issues": ";".join(issues["structural"]),
                "environment_issues": ";".join(issues["environment"]),
                "arithmetic_consistent": not any(
                    issue.startswith(("unit_price_payment_mismatch", "reported_"))
                    for issue in issues["structural"]
                ),
            }
        )
    return row, offer_rows


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_as_float(row.get(field)) for row in rows]
    usable = [value for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _stdev(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_as_float(row.get(field)) for row in rows]
    usable = [value for value in values if value is not None]
    return statistics.stdev(usable) if len(usable) > 1 else None


def _rate(rows: list[dict[str, Any]], field: str, *, eligible: Callable[[dict[str, Any]], bool] | None = None) -> float | None:
    selected = [row for row in rows if eligible is None or eligible(row)]
    values = [row.get(field) for row in selected if row.get(field) is not None]
    return statistics.fmean(1.0 if bool(value) else 0.0 for value in values) if values else None


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, items in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        accepted = [item for item in items if item["accepted"]]
        output.append(
            {
                **dict(zip(keys, values)),
                "n_runs": len(items),
                "n_valid": sum(item["valid"] for item in items),
                "n_accepted": len(accepted),
                "agreement_rate": _rate(items, "accepted"),
                "mean_turns": _mean(items, "turns_taken"),
                "mean_turns_accepted": _mean(accepted, "turns_taken"),
                "mean_offer_count": _mean(items, "offer_count"),
                "proposal_structural_consistency_rate": _mean(
                    items, "proposal_structural_consistency_rate"
                ),
                "final_structural_consistency_rate": _rate(
                    accepted, "contract_structurally_consistent"
                ),
                "final_environment_feasibility_rate": _rate(
                    accepted, "contract_environment_feasible"
                ),
                "final_arithmetic_consistency_rate": _rate(
                    accepted, "arithmetic_consistent"
                ),
                "customer_nominal_ir_rate": _rate(
                    accepted, "customer_nominal_individually_rational"
                ),
                "mean_customer_nominal_surplus": _mean(
                    accepted, "customer_nominal_surplus_vs_batna"
                ),
                "mean_scheduled_payment": _mean(accepted, "scheduled_payment_total"),
                "mean_quantity_A": _mean(accepted, "quantity_A"),
                "mean_quantity_B": _mean(accepted, "quantity_B"),
                "mean_quantity_C": _mean(accepted, "quantity_C"),
                "mean_soup_equivalent_quantity": _mean(
                    accepted, "quantity_soup_equivalents"
                ),
                "mean_week1_payment_share": _mean(accepted, "week1_payment_share"),
                "mean_final_P_sat": _mean(accepted, "P_sat"),
                "std_final_P_sat": _stdev(accepted, "P_sat"),
                "mean_final_completeness": _mean(accepted, "completeness"),
                "std_final_completeness": _stdev(accepted, "completeness"),
                "mean_final_p_ever_broke": _mean(accepted, "p_ever_broke"),
                "mean_E_customer_utility": _mean(accepted, "E_customer_utility"),
                "mean_E_caterer_utility": _mean(accepted, "E_caterer_utility"),
                "jointly_positive_rate": _rate(accepted, "jointly_positive"),
                "customer_acceptance_regret_rate": _rate(
                    accepted, "customer_acceptance_regret"
                ),
                "caterer_acceptance_regret_rate": _rate(
                    accepted, "caterer_acceptance_regret"
                ),
                "strict_json_response_rate": (
                    sum(item["strict_json_count"] for item in items)
                    / sum(item["trace_count"] for item in items)
                    if sum(item["trace_count"] for item in items)
                    else None
                ),
            }
        )
    return output


def role_provider_rows(
    rows: list[dict[str, Any]], offer_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in ("customer", "caterer"):
        remapped = [
            dict(
                row,
                provider=row[f"{role}_provider"],
                expected_gain=row.get(f"{role}_expected_gain"),
                acceptance_regret=row.get(f"{role}_acceptance_regret"),
            )
            for row in rows
        ]
        for item in aggregate(remapped, ("provider",)):
            provider = item["provider"]
            provider_rows = [row for row in remapped if row["provider"] == provider]
            accepted = [row for row in provider_rows if row["accepted"]]
            proposals = [
                offer for offer in offer_rows
                if offer["proposer_role"] == role
                and offer["proposer_provider"] == provider
            ]
            output.append({
                "role": role,
                **item,
                "expected_gain_mean": _mean(accepted, "expected_gain"),
                "expected_gain_std": _stdev(accepted, "expected_gain"),
                "p_sat_mean": _mean(accepted, "P_sat"),
                "p_sat_std": _stdev(accepted, "P_sat"),
                "proposal_feasibility_rate": _rate(proposals, "paper_feasible"),
                "n_proposals": len(proposals),
                "n_acceptance_regret_evaluations": len(accepted),
                "acceptance_regret_rate": _rate(accepted, "acceptance_regret"),
                "contingency_count_mean": _mean(accepted, "contingency_count"),
                "contingency_count_std": _stdev(accepted, "contingency_count"),
                "contingency_rate": (
                    _mean(accepted, "contingency_count") / 3.0
                    if _mean(accepted, "contingency_count") is not None else None
                ),
                "completeness_mean": _mean(accepted, "completeness"),
                "completeness_std": _stdev(accepted, "completeness"),
            })
    return output


def analyze_all(
    runs_root: Path,
    env_root: Path,
    campaign_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matchup_dirs = discover_matchups(runs_root)
    campaign_path = (
        campaign_root or campaign_inferences_root(runs_root.name)
    ) / "campaign.json"
    if campaign_path.is_file():
        campaign = read_json(campaign_path)
    else:
        discovered_ids: dict[int, int] = {}
        for matchup_dir in matchup_dirs:
            canonical_id = _canonical_environment_id(matchup_dir)
            source_id = int(
                read_json(matchup_dir / "data" / "summary.json")["environment_id"]
            )
            previous_id = discovered_ids.setdefault(canonical_id, source_id)
            if previous_id != source_id:
                raise ValueError(
                    f"Conflicting stored environment IDs under {matchup_dir.parents[1]}"
                )
        if any(canonical != source for canonical, source in discovered_ids.items()):
            raise FileNotFoundError(
                f"Missing explicit legacy campaign mapping at {campaign_path}"
            )
        campaign = {
            "schema_version": 1,
            "campaign_id": runs_root.name,
            "source_environment_ids_by_canonical": {
                str(canonical): source
                for canonical, source in sorted(discovered_ids.items())
            },
        }
        write_json(campaign_path, campaign)
    source_ids = {
        int(canonical): int(source)
        for canonical, source in campaign[
            "source_environment_ids_by_canonical"
        ].items()
    }
    fact_cache: dict[int, EnvironmentFacts] = {}
    rows: list[dict[str, Any]] = []
    offer_rows: list[dict[str, Any]] = []
    for matchup_dir in matchup_dirs:
        summary = read_json(matchup_dir / "data" / "summary.json")
        environment_id = _canonical_environment_id(matchup_dir)
        source_environment_id = int(summary["environment_id"])
        if source_ids.get(environment_id) != source_environment_id:
            raise ValueError(
                f"Environment mapping mismatch for {matchup_dir}: canonical "
                f"{environment_id}, stored {source_environment_id}"
            )
        facts = fact_cache.setdefault(
            environment_id,
            environment_facts(environment_id, env_root),
        )
        row, offers = analyze_matchup(matchup_dir, facts)
        rows.append(row)
        offer_rows.extend(offers)
    return rows, offer_rows


def add_contract_quality(
    rows: list[dict[str, Any]],
    offer_rows: list[dict[str, Any]],
    contract_qual_root: Path,
) -> dict[str, Any]:
    """Join precomputed solver results to negotiations and formal offers."""
    records_path = contract_qual_root / "records.jsonl"
    jobs_path = contract_qual_root / "jobs.jsonl"
    manifest_path = contract_qual_root / "manifest.json"
    for path in (records_path, jobs_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing contract-quality artifact: {path}")

    records = read_jsonl(records_path)
    jobs = {
        str(job["solve_hash"]): job
        for job in read_jsonl(jobs_path)
    }
    results: dict[str, dict[str, Any]] = {}
    completeness: dict[str, dict[str, Any]] = {}

    def run_key(record: dict[str, Any]) -> str:
        match = re.fullmatch(r"run[ -](\d+)", str(record["run"]))
        if match is None:
            raise ValueError(f"Unexpected contract-quality run: {record['run']}")
        return (
            f"env-{int(record['environment_id']):02d}/run-{int(match.group(1))}/"
            f"{record['matchup']}"
        )

    def metrics(record: dict[str, Any], *, include_completeness: bool) -> dict[str, Any]:
        solve_hash = record.get("solve_hash")
        job = jobs.get(str(solve_hash), {})
        result: dict[str, Any] = {}
        if solve_hash and record.get("result_path"):
            result_path = contract_qual_root / str(record["result_path"])
            if result_path.is_file():
                if str(solve_hash) not in results:
                    results[str(solve_hash)] = read_json(result_path)
                result = results[str(solve_hash)]
        output = {
            "contract_quality_record_id": record.get("record_id"),
            "contract_quality_offer_index": record.get("offer_index"),
            "solve_hash": solve_hash,
            "solver_status": record.get("solver_status"),
            "contingency_combination": (job.get("contingency") or {}).get("combination"),
            "P_sat": result.get("P_sat"),
            "E_customer_utility": result.get("E_customer_utility"),
            "E_caterer_utility": result.get("E_caterer_utility"),
        }
        if include_completeness:
            metric: dict[str, Any] = {}
            metric_path = record.get("completeness_path")
            if solve_hash and metric_path:
                path = contract_qual_root / str(metric_path)
                if path.is_file():
                    if str(solve_hash) not in completeness:
                        completeness[str(solve_hash)] = read_json(path)
                    metric = completeness[str(solve_hash)]
            output.update({
                "policy_path": record.get("policy_path"),
                "completeness_path": metric_path,
                "completeness": metric.get("completeness"),
                "incompleteness": metric.get("incompleteness"),
                "p_ever_broke": metric.get("p_ever_broke"),
                "survived_but_unsatisfied": metric.get("survived_but_unsatisfied"),
                "survival_curve": metric.get("survival_curve"),
                "incompleteness_curve": metric.get("incompleteness_curve"),
            })
        return output

    offer_records: dict[tuple[str, int], dict[str, Any]] = {}
    final_records: dict[str, dict[str, Any]] = {}
    for record in records:
        key = run_key(record)
        if record.get("is_accepted_final"):
            final_records[key] = record
        if record.get("offer_index") is not None:
            offer_records[(key, int(record["offer_index"]))] = record

    rows_by_run = {str(row["run_key"]): row for row in rows}
    offers_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer in offer_rows:
        record = offer_records.get((str(offer["run_key"]), int(offer["offer_index"])))
        if record is not None:
            offer.update(metrics(record, include_completeness=False))
        run = rows_by_run[str(offer["run_key"])]
        add_utility_metrics(offer, float(run["customer_outside_option"]))
        offers_by_run[str(offer["run_key"])].append(offer)

    for row in rows:
        final = final_records.get(str(row["run_key"]))
        if final is not None:
            row.update(metrics(final, include_completeness=True))
            add_utility_metrics(row, float(row["customer_outside_option"]))
            final_offer = int(row["contract_quality_offer_index"])
            prior = offers_by_run[str(row["run_key"])]
            for role, utility_field in (
                ("customer", "E_customer_utility"),
                ("caterer", "E_caterer_utility"),
            ):
                counterparty = "caterer" if role == "customer" else "customer"
                alternatives = [
                    offer[utility_field]
                    for offer in prior
                    if int(offer["offer_index"]) < final_offer
                    and offer["proposer_role"] == counterparty
                    and offer["paper_feasible"]
                    and offer.get(utility_field) is not None
                ]
                final_utility = row.get(utility_field)
                row[f"{role}_acceptance_regret"] = (
                    any(float(value) > float(final_utility) + 1e-9 for value in alternatives)
                    if final_utility is not None
                    else None
                )

    manifest = read_json(manifest_path)
    return {
        **manifest,
        "n_offers_joined": sum(offer.get("P_sat") is not None for offer in offer_rows),
        "n_finals_joined": sum(row.get("P_sat") is not None for row in rows if row["accepted"]),
        "n_completeness_joined": sum(
            row.get("completeness") is not None for row in rows if row["accepted"]
        ),
    }


def add_utility_metrics(row: dict[str, Any], customer_batna: float) -> None:
    customer = row.get("E_customer_utility")
    caterer = row.get("E_caterer_utility")
    if customer is None or caterer is None:
        return
    row["customer_expected_gain"] = float(customer) - customer_batna
    row["caterer_expected_gain"] = float(caterer)
    row["customer_positive_vs_batna"] = float(customer) >= customer_batna - 1e-9
    row["caterer_positive_vs_batna"] = float(caterer) >= -1e-9
    row["jointly_positive"] = (
        row["customer_positive_vs_batna"]
        and row["caterer_positive_vs_batna"]
    )


def build_summary(
    rows: list[dict[str, Any]],
    offer_rows: list[dict[str, Any]],
    contract_quality: dict[str, Any],
) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    issue_counts = Counter()
    for row in accepted:
        for issue in filter(None, str(row.get("structural_issues") or "").split(";")):
            issue_counts[issue.split(":", 1)[0]] += 1
        for issue in filter(None, str(row.get("environment_issues") or "").split(";")):
            issue_counts[issue.split(":", 1)[0]] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_semantics": {
            "customer_nominal_utility": "Starting budget - scheduled payments + private value of all promised dishes; assumes full performance.",
            "customer_nominal_surplus_vs_batna": "Nominal utility minus the environment's customer outside-option utility.",
            "structural_consistency": "All required terms parse and unit-price, reported-total, and payment arithmetic agree.",
            "environment_feasibility": "Scheduled payments respect the customer budget, minimum dishes are met, and no production round exceeds the pre-production inventory cap.",
            "proposal_structural_consistency_rate": "Fraction of all formal offers whose required terms and internal arithmetic are coherent; model errors remain failures.",
            "P_sat": "Probability that the contract reaches the solver's satisfaction threshold under the environment model and optimal policy.",
            "expected_utility": "Environment-aware expected utility under the solver's selected policy; customer gain is relative to its outside option and caterer gain is relative to zero.",
            "completeness": "Mean probability of remaining in a live policy state across production rounds; calculated for accepted final agreements only.",
            "p_ever_broke": "Probability that an accepted final agreement reaches the absorbing dead state by the end of production.",
        },
        "n_runs": len(rows),
        "n_valid": sum(row["valid"] for row in rows),
        "n_accepted": len(accepted),
        "n_no_agreement": len(rows) - len(accepted),
        "agreement_rate": len(accepted) / len(rows) if rows else None,
        "n_final_structural_failures": sum(
            not row.get("contract_structurally_consistent", False) for row in accepted
        ),
        "n_final_arithmetic_failures": sum(
            not row.get("arithmetic_consistent", False) for row in accepted
        ),
        "n_final_environment_failures": sum(
            not row.get("contract_environment_feasible", False) for row in accepted
        ),
        "n_formal_offers": len(offer_rows),
        "formal_offer_structural_consistency_rate": _rate(
            offer_rows, "structurally_consistent"
        ),
        "trace_response_count": sum(row["trace_count"] for row in rows),
        "strict_json_response_count": sum(row["strict_json_count"] for row in rows),
        "recovered_json_response_count": sum(row["recovered_json_count"] for row in rows),
        "plain_text_fallback_count": sum(row["plain_text_fallback_count"] for row in rows),
        "n_formal_offers_with_P_sat": sum(
            offer.get("P_sat") is not None for offer in offer_rows
        ),
        "formal_offer_P_sat_coverage": (
            sum(offer.get("P_sat") is not None for offer in offer_rows) / len(offer_rows)
            if offer_rows
            else None
        ),
        "mean_formal_offer_P_sat": _mean(offer_rows, "P_sat"),
        "n_final_contracts_with_P_sat": sum(
            row.get("P_sat") is not None for row in accepted
        ),
        "final_P_sat_coverage": (
            sum(row.get("P_sat") is not None for row in accepted) / len(accepted)
            if accepted
            else None
        ),
        "mean_final_P_sat": _mean(accepted, "P_sat"),
        "std_final_P_sat": _stdev(accepted, "P_sat"),
        "final_P_sat_ge_0_95_rate": _rate(
            [dict(row, meets_P_sat=float(row["P_sat"]) >= 0.95) for row in accepted if row.get("P_sat") is not None],
            "meets_P_sat",
        ),
        "mean_final_E_customer_utility": _mean(accepted, "E_customer_utility"),
        "std_final_E_customer_utility": _stdev(accepted, "E_customer_utility"),
        "mean_final_E_caterer_utility": _mean(accepted, "E_caterer_utility"),
        "std_final_E_caterer_utility": _stdev(accepted, "E_caterer_utility"),
        "accepted_jointly_positive_rate": _rate(accepted, "jointly_positive"),
        "customer_acceptance_regret_rate": _rate(
            accepted, "customer_acceptance_regret"
        ),
        "caterer_acceptance_regret_rate": _rate(
            accepted, "caterer_acceptance_regret"
        ),
        "n_final_contracts_with_completeness": sum(
            row.get("completeness") is not None for row in accepted
        ),
        "final_completeness_coverage": (
            sum(row.get("completeness") is not None for row in accepted) / len(accepted)
            if accepted
            else None
        ),
        "mean_final_completeness": _mean(accepted, "completeness"),
        "std_final_completeness": _stdev(accepted, "completeness"),
        "mean_final_p_ever_broke": _mean(accepted, "p_ever_broke"),
        "contract_quality": contract_quality,
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--contract-qual", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs_root = (args.runs_root or campaign_runs_root(args.campaign)).resolve()
    inference_root = campaign_inferences_root(args.campaign)
    rows, offer_rows = analyze_all(
        runs_root,
        args.env_root.resolve(),
        campaign_root=inference_root,
    )
    if not rows:
        raise SystemExit(f"No negotiation summaries found under {runs_root}")

    contract_qual_root = (
        args.contract_qual or inference_root / "contract-qual"
    ).resolve()
    contract_quality = add_contract_quality(rows, offer_rows, contract_qual_root)
    output = (args.output or inference_root / "stats.json").resolve()
    summary = build_summary(rows, offer_rows, contract_quality)
    by_environment = aggregate(rows, ("environment_id", "environment_name"))
    by_matchup = aggregate(rows, ("matchup", "customer_provider", "caterer_provider"))
    by_environment_matchup = aggregate(rows, ("environment_id", "matchup"))
    by_role_provider = role_provider_rows(rows, offer_rows)
    issues = [
        row
        for row in rows
        if row.get("structural_issues") or row.get("environment_issues")
    ]

    write_json(output, {
        "schema_version": 1,
        "campaign": args.campaign,
        "summary": summary,
        "negotiations": rows,
        "offers": offer_rows,
        "issues": issues,
        "by_environment": by_environment,
        "by_matchup": by_matchup,
        "by_environment_matchup": by_environment_matchup,
        "by_role_provider": by_role_provider,
    })
    print(
        f"Analyzed {summary['n_runs']} negotiations and {summary['n_formal_offers']} formal offers; "
        f"wrote metrics to {output}"
    )


if __name__ == "__main__":
    main()
