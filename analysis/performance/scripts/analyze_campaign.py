"""Summarize a RatConBench performance campaign into CSV and JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from analysis.performance.metrics import analyze_run


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "performance-main"
DEFAULT_CONTRACTS_ROOT = (
    PROJECT_ROOT / "algorithms" / "policy-suite-output-check" / "policy-suite"
)
INFERENCES_ROOT = Path(__file__).resolve().parents[1] / "inferences"


def _default_output_dir(campaign: Path) -> Path:
    """Route analyzed campaigns into their main or ablation inference arm."""
    source_to_arm = {
        "performance-main": "main",
        "performance-abl": "abl",
    }
    try:
        arm = source_to_arm[campaign.parent.name]
    except KeyError as exc:
        raise ValueError(
            "cannot infer output arm from campaign path; pass --output-dir"
        ) from exc
    return INFERENCES_ROOT / arm / campaign.name


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _latest_campaign() -> Path:
    campaigns = sorted(path for path in RESULTS_ROOT.iterdir() if path.is_dir())
    if not campaigns:
        raise FileNotFoundError(f"No campaigns found under {RESULTS_ROOT}")
    return campaigns[-1]


def _contract_dir(summary: dict[str, Any], contracts_root: Path) -> Path:
    saved_text = summary.get("contract_dir")
    if saved_text:
        saved = Path(str(saved_text))
        if saved.is_dir():
            return saved
    fallback = contracts_root / str(summary["contract_id"])
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"Cannot find contract {summary['contract_id']}")


def _mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(mean(numbers), 6) if numbers else None


def _aggregate(rows: list[dict[str, Any]], metric_fields: list[str]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        **{field: _mean([row.get(field) for row in rows]) for field in metric_fields},
    }


def _aggregate_cases(
    rows: list[dict[str, Any]], metric_fields: list[str]
) -> dict[str, Any]:
    """Average seeds within a case, then summarize across cases."""
    samples: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        samples[(row["environment_id"], row["contract_id"])].append(row)

    result: dict[str, Any] = {"runs": len(rows), "cases": len(samples)}
    for field in metric_fields:
        values = [
            sample_mean
            for sample_rows in samples.values()
            if (sample_mean := _mean([row.get(field) for row in sample_rows]))
            is not None
        ]
        result[field] = round(mean(values), 6) if values else None
        result[f"{field}_std"] = (
            round(stdev(values), 6)
            if len(values) > 1
            else (0.0 if values else None)
        )
    return result


def _add_regret_metrics(rows: list[dict[str, Any]]) -> None:
    """Compare each role with its matching same-seed RCC run."""
    runs = {
        (
            row["environment_id"],
            row["contract_id"],
            row["seed"],
            row["customer"],
            row["supplier"],
        ): row
        for row in rows
        if row["valid"]
    }
    for row in rows:
        if not row["valid"]:
            continue
        common = (row["environment_id"], row["contract_id"], row["seed"])
        references = {
            "customer": runs.get((*common, "rcc", row["supplier"])),
            "supplier": runs.get((*common, row["customer"], "rcc")),
        }
        for role, reference in references.items():
            raw_field = f"{role}_ex_post_regret"
            compliant_field = f"{role}_ex_post_compliant_regret_zeroed"
            if reference is None:
                row[raw_field] = None
                row[compliant_field] = None
                continue
            row[raw_field] = round(
                reference[f"{role}_utility"] - row[f"{role}_utility"], 6
            )
            row[compliant_field] = round(
                reference[f"{role}_compliant_utility_zeroed"]
                - row[f"{role}_compliant_utility_zeroed"],
                6,
            )


def _aggregate_by_role(
    rows: list[dict[str, Any]], metric_fields: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "customer": defaultdict(list),
        "supplier": defaultdict(list),
    }
    for row in rows:
        groups["customer"][
            f"focal-{row['customer']}_counterparty-{row['supplier']}"
        ].append(row)
        groups["supplier"][
            f"focal-{row['supplier']}_counterparty-{row['customer']}"
        ].append(row)

    fields = {
        "customer": [field for field in metric_fields if field.startswith("customer_")],
        "supplier": [
            field
            for field in metric_fields
            if field.startswith("supplier_") or field == "delivery_shortfalls"
        ],
    }
    return {
        role: {
            matchup: _aggregate_cases(group_rows, fields[role])
            for matchup, group_rows in sorted(role_groups.items())
        }
        for role, role_groups in groups.items()
    }


def analyze_campaign(
    campaign: Path,
    contracts_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(campaign.rglob("data/summary.json")):
        saved = _read_json(summary_path)
        relative_run = summary_path.parents[1].relative_to(campaign)
        base = {
            "run_id": str(relative_run),
            "contract_id": saved.get("contract_id"),
            "environment_id": saved.get("environment_id"),
            "environment_name": saved.get("environment_name"),
            "customer": saved.get("customer"),
            "supplier": saved.get("supplier"),
            "seed": saved.get("seed"),
            "valid": bool(saved.get("valid", False)),
            "invalid_reason": saved.get("invalid_reason"),
        }
        if not base["valid"]:
            rows.append(base)
            continue
        contract_dir = _contract_dir(saved, contracts_root)
        config = _read_json(contract_dir / "config.json")
        terms = _read_json(contract_dir / "terms.json")
        rows.append(
            {
                **base,
                "contingency_variant": config.get("contingency_variant"),
                "source_band": config.get("source_band"),
                **analyze_run(saved, terms, config),
            }
        )

    _add_regret_metrics(rows)
    valid_rows = [row for row in rows if row["valid"]]
    identity_fields = {
        "run_id", "contract_id", "environment_id", "environment_name",
        "customer", "supplier", "seed", "valid", "invalid_reason",
        "contingency_variant", "source_band",
    }
    metric_fields = sorted(
        {key for row in valid_rows for key in row} - identity_fields
    )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_environment": defaultdict(list),
        "by_contingency": defaultdict(list),
        "by_source_band": defaultdict(list),
    }
    for row in valid_rows:
        grouped["by_environment"][str(row["environment_id"])].append(row)
        grouped["by_contingency"][str(row["contingency_variant"])].append(row)
        grouped["by_source_band"][str(row["source_band"])].append(row)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign),
        "runs": len(rows),
        "valid_runs": len(valid_rows),
        "invalid_runs": len(rows) - len(valid_rows),
        "overall": _aggregate(valid_rows, metric_fields),
        "by_role": _aggregate_by_role(valid_rows, metric_fields),
        **{
            name: {
                key: _aggregate(group_rows, metric_fields)
                for key, group_rows in sorted(groups.items())
            }
            for name, groups in grouped.items()
        },
    }
    return rows, report


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--latest", action="store_true")
    parser.add_argument("--contracts-root", type=Path, default=DEFAULT_CONTRACTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    campaign = (_latest_campaign() if args.latest else args.run_dir).resolve()
    if not campaign.is_dir():
        parser.error(f"campaign does not exist: {campaign}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else _default_output_dir(campaign)
    )
    rows, report = analyze_campaign(campaign, args.contracts_root.resolve())
    _write_outputs(output_dir, rows, report)
    print(f"Analyzed {report['valid_runs']}/{report['runs']} runs")
    print(f"Wrote {output_dir / 'rows.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
