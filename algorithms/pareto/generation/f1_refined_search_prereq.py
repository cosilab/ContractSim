#!/usr/bin/env python3
"""Select accepted LLM contracts that are not dominated by the F1 frontier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    algorithms_root = Path(__file__).resolve().parents[2]
    if str(algorithms_root) not in sys.path:
        sys.path.insert(0, str(algorithms_root))

from pareto.generation.f1_common import save_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[3]
PARETO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_QUAL_ROOT = (
    REPO_ROOT
    / "analysis"
    / "negotiation"
    / "inferences"
    / "main"
    / "control"
    / "contract-qual"
)
DEFAULT_RESULTS_ROOT = PARETO_ROOT / "data"
P_SAT_TOLERANCE = 1e-12


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def dominates(
    frontier_point: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Return whether one F1 point strictly Pareto-dominates a candidate."""
    frontier_supplier = _finite_number(
        frontier_point["E_supplier_utility"], name="frontier supplier utility"
    )
    frontier_customer = _finite_number(
        frontier_point["E_customer_utility"], name="frontier customer utility"
    )
    candidate_supplier = _finite_number(
        candidate["E_supplier_utility"], name="candidate supplier utility"
    )
    candidate_customer = _finite_number(
        candidate["E_customer_utility"], name="candidate customer utility"
    )
    return (
        frontier_supplier >= candidate_supplier
        and frontier_customer >= candidate_customer
        and (
            frontier_supplier > candidate_supplier
            or frontier_customer > candidate_customer
        )
    )


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL input: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_environment(
    environment_id: int,
    *,
    contract_qual_root: Path,
    results_root: Path,
    p_sat_floor: float = 0.95,
) -> dict[str, Any]:
    """Build one environment's deterministic refined-anchor manifest."""
    frontier_path = (
        results_root
        / f"env-{environment_id:02d}"
        / "f1-merged"
        / "f1_frontier.json"
    )
    if not frontier_path.is_file():
        raise FileNotFoundError(
            f"Missing F1 frontier for environment {environment_id}: {frontier_path}"
        )
    frontier_payload = json.loads(frontier_path.read_text())
    frontier_points = [
        point
        for point in frontier_payload.get("points", [])
        if _finite_number(point["P_sat"], name="frontier P_sat")
        + P_SAT_TOLERANCE
        >= p_sat_floor
    ]
    if not frontier_points:
        raise ValueError(f"Environment {environment_id} has no eligible F1 points")

    jobs = _load_json_lines(contract_qual_root / "jobs.jsonl")
    counts = {
        "accepted_jobs": 0,
        "unsupported_contingency": 0,
        "missing_or_failed_solve": 0,
        "below_satisfaction_floor": 0,
        "dominated": 0,
    }
    anchors: list[dict[str, Any]] = []

    for job in jobs:
        if int(job.get("environment_id", -1)) != environment_id:
            continue
        accepted_record_ids = list(job.get("accepted_final_record_ids", []))
        if not accepted_record_ids:
            continue
        counts["accepted_jobs"] += 1
        if job.get("contingency", {}).get("combination") != "grim_trigger":
            counts["unsupported_contingency"] += 1
            continue

        solve_hash = str(job["solve_hash"])
        solve_path = (
            contract_qual_root
            / "solves"
            / f"env-{environment_id:02d}"
            / f"{solve_hash}.json"
        )
        if not solve_path.is_file():
            counts["missing_or_failed_solve"] += 1
            continue
        solve = json.loads(solve_path.read_text())
        if solve.get("solver_status") != "solved":
            counts["missing_or_failed_solve"] += 1
            continue

        candidate = {
            "E_supplier_utility": _finite_number(
                solve["E_caterer_utility"], name="LLM supplier utility"
            ),
            "E_customer_utility": _finite_number(
                solve["E_customer_utility"], name="LLM customer utility"
            ),
            "P_sat": _finite_number(solve["P_sat"], name="LLM P_sat"),
        }
        if candidate["P_sat"] + P_SAT_TOLERANCE < p_sat_floor:
            counts["below_satisfaction_floor"] += 1
            continue
        if any(dominates(point, candidate) for point in frontier_points):
            counts["dominated"] += 1
            continue

        anchors.append(
            {
                "anchor_id": solve_hash,
                "solve_hash": solve_hash,
                "accepted_final_record_ids": sorted(accepted_record_ids),
                "contract": job["contract"],
                **candidate,
            }
        )

    anchors.sort(key=lambda anchor: anchor["anchor_id"])
    return {
        "schema_version": 1,
        "environment_id": environment_id,
        "environment_name": frontier_payload.get("environment_name"),
        "environment_fingerprint": frontier_payload.get("environment_fingerprint"),
        "p_sat_floor": p_sat_floor,
        "contract_qual_root": str(contract_qual_root.resolve()),
        "f1_frontier": str(frontier_path.resolve()),
        "selection_counts": counts,
        "anchor_count": len(anchors),
        "anchors": anchors,
    }


def _discover_environments(results_root: Path) -> list[int]:
    environments: list[int] = []
    for path in results_root.glob("env-*/f1-merged/f1_frontier.json"):
        try:
            environments.append(int(path.parents[1].name.removeprefix("env-")))
        except ValueError:
            continue
    return sorted(set(environments))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        action="append",
        type=int,
        help="Environment ID to prepare; repeat for multiple IDs (default: discover)",
    )
    parser.add_argument(
        "--contract-qual-root",
        type=Path,
        default=DEFAULT_CONTRACT_QUAL_ROOT,
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--p-sat-floor", type=float, default=0.95)
    args = parser.parse_args()

    contract_qual_root = args.contract_qual_root.resolve()
    results_root = args.results_root.resolve()
    environment_ids = args.env or _discover_environments(results_root)
    if not environment_ids:
        parser.error("No environments with f1_frontier.json were found")

    total = 0
    for environment_id in environment_ids:
        payload = prepare_environment(
            environment_id,
            contract_qual_root=contract_qual_root,
            results_root=results_root,
            p_sat_floor=args.p_sat_floor,
        )
        output = (
            results_root
            / f"env-{environment_id:02d}"
            / "f1-refinement"
            / "anchors.json"
        )
        save_json_atomic(output, payload)
        total += payload["anchor_count"]
        print(
            f"env-{environment_id:02d}: saved {payload['anchor_count']} anchors to "
            f"{output}",
            flush=True,
        )
    print(f"Saved {total} refined-search anchors", flush=True)


if __name__ == "__main__":
    main()
