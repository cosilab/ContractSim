#!/usr/bin/env python3
"""Prepare all saved domain-robustness performance results for analysis.

The robustness runner stores results as ``<domain>/<role>/<contract>/<run>``
instead of the timestamped campaign layout used by the main performance
runner.  This command presents each result domain in the same compact
``rows.csv`` and ``summary.json`` format used by the existing performance
analysis tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.performance.metrics import analyze_run  # noqa: E402
from analysis.performance.scripts.analyze_campaign import (  # noqa: E402
    _aggregate,
    _aggregate_by_role,
    _contract_dir,
    _read_json,
    analyze_campaign,
)


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "performance-domain-robustness"
DEFAULT_BASELINE_RESULTS_ROOT = DEFAULT_RESULTS_ROOT / "rational baselines"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis" / "performance" / "inferences" / "domain-robustness"
)
DEFAULT_CONTRACTS_ROOT = (
    PROJECT_ROOT / "algorithms" / "synth-contracts" / "data" / "suite"
)


def _load_main_env5_baselines(
    results_root: Path,
    contracts_root: Path,
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    """Load Env-5 baseline rows used as RCC regret references."""
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Main performance results do not exist or are not a directory: {results_root}"
        )

    references: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for summary_path in sorted(results_root.rglob("data/summary.json")):
        saved = _read_json(summary_path)
        if (
            not saved.get("valid")
            or saved.get("environment_id") != 5
            or saved.get("customer") not in {"rc", "re", "rcc"}
            or saved.get("supplier") not in {"rc", "re", "rcc"}
            or "rcc" not in {saved.get("customer"), saved.get("supplier")}
        ):
            continue

        try:
            contract_dir = _contract_dir(saved, contracts_root)
        except FileNotFoundError:
            # Ignore stale legacy copies under the main results root.
            continue
        config = _read_json(contract_dir / "config.json")
        terms = _read_json(contract_dir / "terms.json")
        key = (
            str(saved["contract_id"]),
            int(saved["seed"]),
            str(saved["customer"]),
            str(saved["supplier"]),
        )
        # The main results contain duplicate copies of these deterministic
        # baseline runs; one row per contract/matchup/seed is enough here.
        references.setdefault(key, analyze_run(saved, terms, config))

    if not references:
        raise FileNotFoundError(
            f"No valid Env-5 baseline/RCC reference runs found under {results_root}"
        )
    return references


def _add_main_baseline_regrets(
    rows: list[dict[str, Any]],
    references: dict[tuple[str, int, str, str], dict[str, Any]],
) -> dict[str, int]:
    """Fill regret fields using the main Env-5 RCC baseline runs."""
    matches = {"customer": 0, "supplier": 0}
    for row in rows:
        if not row["valid"]:
            continue
        for role in ("customer", "supplier"):
            raw_field = f"{role}_ex_post_regret"
            compliant_field = f"{role}_ex_post_compliant_regret_zeroed"
            if row[role] == "rcc":
                reference = row
            else:
                matchup = (
                    ("rcc", row["supplier"])
                    if role == "customer"
                    else (row["customer"], "rcc")
                )
                reference = references.get(
                    (str(row["contract_id"]), int(row["seed"]), *matchup)
                )
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
            matches[role] += 1
    return matches


def _refresh_regret_aggregates(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    """Refresh report aggregates after adding cross-campaign regrets."""
    valid_rows = [row for row in rows if row["valid"]]
    fields = [
        "customer_ex_post_regret",
        "customer_ex_post_compliant_regret_zeroed",
        "supplier_ex_post_regret",
        "supplier_ex_post_compliant_regret_zeroed",
    ]
    report["overall"].update(_aggregate(valid_rows, fields))
    for role, groups in _aggregate_by_role(valid_rows, fields).items():
        for matchup, values in groups.items():
            report["by_role"][role][matchup].update(values)

    for report_name, row_field in (
        ("by_environment", "environment_id"),
        ("by_contingency", "contingency_variant"),
        ("by_source_band", "source_band"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_rows:
            grouped[str(row[row_field])].append(row)
        for key, group_rows in grouped.items():
            report[report_name][key].update(_aggregate(group_rows, fields))


def discover_domains(results_root: Path) -> dict[str, Path]:
    """Return result directories that contain at least one saved summary."""
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Results root does not exist or is not a directory: {results_root}"
        )

    domains = {
        path.name: path
        for path in sorted(results_root.iterdir())
        if (
            path.is_dir()
            and path.name != "rational baselines"
            and any(path.rglob("data/summary.json"))
        )
    }
    if not domains:
        raise FileNotFoundError(
            f"No result domains containing data/summary.json found under {results_root}"
        )
    return domains


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


def prepare_domain(
    domain: str,
    source_dir: Path,
    output_root: Path,
    contracts_root: Path,
    baseline_references: dict[tuple[str, int, str, str], dict[str, Any]],
    baseline_results_root: Path,
) -> dict[str, Any]:
    """Analyze one result domain and write its analysis-ready artifacts."""
    rows, report = analyze_campaign(source_dir, contracts_root)
    regret_matches = _add_main_baseline_regrets(rows, baseline_references)
    _refresh_regret_aggregates(rows, report)
    rows = [{"domain": domain, **row} for row in rows]
    report = {
        **report,
        "domain": domain,
        "source_dir": str(source_dir),
        "regret_reference_root": str(baseline_results_root),
        "regret_reference_runs": len(baseline_references),
        "regret_matches": regret_matches,
    }
    output_dir = output_root / domain
    _write_outputs(output_dir, rows, report)
    return report


def prepare_all(
    *,
    results_root: Path,
    output_root: Path,
    contracts_root: Path,
    baseline_results_root: Path,
    selected_domains: list[str] | None = None,
) -> dict[str, Any]:
    """Prepare selected domains and return the generated manifest."""
    available = discover_domains(results_root)
    domains = sorted(selected_domains or available)
    missing = [domain for domain in domains if domain not in available]
    if missing:
        raise FileNotFoundError(
            "Unknown result domain(s): " + ", ".join(missing)
        )

    baseline_references = _load_main_env5_baselines(
        baseline_results_root,
        contracts_root,
    )

    reports: dict[str, dict[str, Any]] = {}
    for domain in domains:
        source_dir = available[domain]
        print(f"Analyzing {domain}: {source_dir}", flush=True)
        reports[domain] = prepare_domain(
            domain,
            source_dir,
            output_root,
            contracts_root,
            baseline_references,
            baseline_results_root,
        )
        report = reports[domain]
        print(
            f"  {report['valid_runs']}/{report['runs']} valid runs -> "
            f"{output_root / domain}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "contracts_root": str(contracts_root),
        "baseline_results_root": str(baseline_results_root),
        "baseline_reference_runs": len(baseline_references),
        "output_root": str(output_root),
        "domains": {
            domain: {
                "source_dir": report["source_dir"],
                "output_dir": str(output_root / domain),
                "runs": report["runs"],
                "valid_runs": report["valid_runs"],
                "invalid_runs": report["invalid_runs"],
            }
            for domain, report in reports.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="root containing one directory per result domain",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory for analysis/performance/inferences artifacts",
    )
    parser.add_argument(
        "--baseline-results-root",
        type=Path,
        default=DEFAULT_BASELINE_RESULTS_ROOT,
        help="Env-5 rational-baseline runs (seeds 42-44) used as RCC regret references",
    )
    parser.add_argument(
        "--contracts-root",
        type=Path,
        default=DEFAULT_CONTRACTS_ROOT,
        help="canonical Env-5 contract suite used by the metric calculator",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="optional subset of result-domain directory names",
    )
    args = parser.parse_args()

    try:
        prepare_all(
            results_root=args.results_root.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
            contracts_root=args.contracts_root.expanduser().resolve(),
            baseline_results_root=args.baseline_results_root.expanduser().resolve(),
            selected_domains=args.domains,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Prepared performance analysis under {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
