#!/usr/bin/env python3
"""Run refined F1 search for every selected anchor in one environment and sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    algorithms_root = Path(__file__).resolve().parents[2]
    if str(algorithms_root) not in sys.path:
        sys.path.insert(0, str(algorithms_root))

from pareto.generation.f1_common import (
    DEFAULT_ENVIRONMENT_ROOT,
    Contract,
    F1RefinedSearchConfig,
    RefinedContractCandidate,
    Sweep,
    file_fingerprint,
    load_environment_selection,
    save_json_atomic,
)
from pareto.generation.f1_refined_search import F1FrontierSearchRefined


REPO_ROOT = Path(__file__).resolve().parents[3]
PARETO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PARETO_ROOT / "data"


def focus_utility(anchor: Mapping[str, Any], sweep: Sweep) -> float:
    return float(
        anchor[
            "E_customer_utility"
            if sweep is Sweep.MAX_CUSTOMER
            else "E_supplier_utility"
        ]
    )


def constrained_utility(anchor: Mapping[str, Any], sweep: Sweep) -> float:
    return float(
        anchor[
            "E_supplier_utility"
            if sweep is Sweep.MAX_CUSTOMER
            else "E_customer_utility"
        ]
    )


def ordered_anchors(
    anchors: list[dict[str, Any]], sweep: Sweep
) -> list[dict[str, Any]]:
    """Sort anchors by increasing focus-role utility with a stable tie-breaker."""
    return sorted(
        anchors,
        key=lambda anchor: (focus_utility(anchor, sweep), str(anchor["anchor_id"])),
    )


def anchor_epsilon_grid(
    anchor: Mapping[str, Any], sweep: Sweep, radius: float
) -> tuple[float, float, float]:
    if radius <= 0:
        raise ValueError("epsilon radius must be positive")
    center = constrained_utility(anchor, sweep)
    return (center - radius, center, center + radius)


def _anchor_point(
    anchor: Mapping[str, Any],
    candidate: RefinedContractCandidate,
    sweep: Sweep,
) -> dict[str, Any]:
    return {
        "epsilon": constrained_utility(anchor, sweep),
        "sweep": sweep.value,
        "epsilon_role": sweep.constrained_role,
        "optimized_role": sweep.optimized_role,
        "candidate": candidate.to_dict(),
        "P_sat": float(anchor["P_sat"]),
        "E_supplier_utility": float(anchor["E_supplier_utility"]),
        "E_customer_utility": float(anchor["E_customer_utility"]),
        "constraint_met": True,
        "score": focus_utility(anchor, sweep),
        "source": "llm_anchor",
        "anchor_id": str(anchor["anchor_id"]),
    }


def run_batch(
    *,
    environment_id: int,
    sweep: Sweep,
    anchors_path: Path,
    results_root: Path,
    environment_root: Path,
    epsilon_radius: float,
    resume: bool,
) -> Path:
    manifest = json.loads(anchors_path.read_text())
    if int(manifest["environment_id"]) != environment_id:
        raise ValueError(
            f"Anchor manifest is for environment {manifest['environment_id']}, "
            f"not {environment_id}"
        )
    anchors = ordered_anchors(list(manifest.get("anchors", [])), sweep)
    if not anchors:
        raise ValueError(f"Anchor manifest contains no anchors: {anchors_path}")

    environment_path, environment = load_environment_selection(
        environment_id=environment_id,
        environment_path=None,
        environment_root=environment_root,
    )
    environment_digest = file_fingerprint(environment_path)
    manifest_digest = manifest.get("environment_fingerprint")
    if manifest_digest and manifest_digest != environment_digest:
        raise ValueError("Anchor manifest and current environment fingerprints differ")

    from rc_solver import RCCSolver

    solver = RCCSolver(environment)
    p_sat_floor = float(manifest.get("p_sat_floor", 0.95))
    refinement_root = results_root / f"env-{environment_id:02d}" / "f1-refinement"
    aggregate_points: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors, 1):
        anchor_id = str(anchor["anchor_id"])
        candidate = RefinedContractCandidate.from_contract(
            Contract.from_mapping(anchor["contract"]), environment.supplier.products
        )
        epsilon_grid = anchor_epsilon_grid(anchor, sweep, epsilon_radius)
        config = F1RefinedSearchConfig(
            sweep=sweep,
            epsilon_grid=epsilon_grid,
            satisfaction_threshold=p_sat_floor,
            minimum_satisfaction=p_sat_floor,
            number_of_starts=1,
            payment_cap=int(environment.customer.starting_budget),
        )
        anchor_root = refinement_root / "anchors" / anchor_id
        checkpoint_path = anchor_root / f"{sweep.value}-f1.checkpoint.json"
        result_path = anchor_root / f"{sweep.value}-f1.json"
        print(
            f"[{index}/{len(anchors)}] {anchor_id[:12]} "
            f"{sweep.optimized_role}={focus_utility(anchor, sweep):.3f}",
            flush=True,
        )
        search = F1FrontierSearchRefined(
            solver,
            environment,
            config,
            candidate,
            checkpoint_path,
            environment_fingerprint=environment_digest,
        )
        points = search.run(resume=resume and checkpoint_path.is_file())
        serialized_points = []
        for point in points:
            row = point.to_dict()
            row.update({"source": "refined", "anchor_id": anchor_id})
            serialized_points.append(row)

        result = search.result_payload(points)
        result.update(
            {
                "anchor_id": anchor_id,
                "accepted_final_record_ids": anchor["accepted_final_record_ids"],
                "anchor_evaluation": {
                    "P_sat": float(anchor["P_sat"]),
                    "E_supplier_utility": float(anchor["E_supplier_utility"]),
                    "E_customer_utility": float(anchor["E_customer_utility"]),
                },
                "points": serialized_points,
            }
        )
        save_json_atomic(result_path, result)
        aggregate_points.append(_anchor_point(anchor, candidate, sweep))
        aggregate_points.extend(serialized_points)

    output_path = refinement_root / f"{sweep.value}-f1.json"
    save_json_atomic(
        output_path,
        {
            "schema_version": 2,
            "frontier": "F1",
            "status": "complete",
            "search_variant": "targeted-refinement-batch",
            "environment_id": environment.environment_id,
            "environment_name": environment.environment_name,
            "environment_axes": dict(environment.axes),
            "environment_fingerprint": environment_digest,
            "sweep": sweep.value,
            "anchors_file": str(anchors_path.resolve()),
            "anchor_count": len(anchors),
            "points": aggregate_points,
        },
    )
    print(f"Saved {len(aggregate_points)} candidates to {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument(
        "--sweep", choices=tuple(sweep.value for sweep in Sweep), required=True
    )
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--environment-root", type=Path, default=DEFAULT_ENVIRONMENT_ROOT
    )
    parser.add_argument("--epsilon-radius", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    anchors_path = (
        args.anchors
        or results_root
        / f"env-{args.env:02d}"
        / "f1-refinement"
        / "anchors.json"
    ).resolve()
    if not anchors_path.is_file():
        parser.error(f"Missing anchor manifest: {anchors_path}")
    run_batch(
        environment_id=args.env,
        sweep=Sweep(args.sweep),
        anchors_path=anchors_path,
        results_root=results_root,
        environment_root=args.environment_root.resolve(),
        epsilon_radius=args.epsilon_radius,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
