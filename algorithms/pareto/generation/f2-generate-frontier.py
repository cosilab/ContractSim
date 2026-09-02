#!/usr/bin/env python3
"""Generate one role's F2 candidates from its base and refined F1 points."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS_ROOT = REPOSITORY_ROOT / "algorithms"
RC_SOLVER_ROOT = ALGORITHMS_ROOT / "rc-solver"
POLICY_GENERATION_ROOT = ALGORITHMS_ROOT / "synth-contracts" / "generation"
DATA_ROOT = ALGORITHMS_ROOT / "pareto" / "data"
for path in (RC_SOLVER_ROOT, POLICY_GENERATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contract import Contract  # noqa: E402
from environment import Environment  # noqa: E402
from generate_policy_suite import (  # noqa: E402
    CONTINGENCY_SPECS,
    _metrics,
    _round_inputs,
    _solver_for,
)
from rc_solver import RCCSolver  # noqa: E402
from runtime import TransitionRuntime  # noqa: E402


P_SAT_FLOOR = 0.95
PRODUCTION_WEEKS = (2, 4, 6, 8, 10)
PAYMENT_WEEKS = (1, 3, 5, 7, 9, 11)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def roles(sweep: str) -> tuple[str, str]:
    if sweep == "max-customer":
        return "supplier", "customer"
    return "customer", "supplier"


def load_role_frontier(
    environment_root: Path, sweep: str
) -> tuple[int, list[dict[str, Any]]]:
    constrained_role, _ = roles(sweep)
    pool = []
    for source in ("f1-base", "f1-refinement"):
        path = environment_root / source / f"{sweep}-f1.json"
        payload = json.loads(path.read_text())
        pool.extend({**point, "f1_source": source} for point in payload["points"])

    eligible = [
        point
        for point in pool
        if point["P_sat"] + 1e-12 >= P_SAT_FLOOR
        and point[f"E_{constrained_role}_utility"] + 1e-12 >= point["epsilon"]
    ]
    unique = {
        (
            json.dumps(point["candidate"], sort_keys=True),
            round(point["epsilon"], 8),
        ): point
        for point in eligible
    }
    frontier = [
        point
        for point in unique.values()
        if not any(
            other["E_supplier_utility"] >= point["E_supplier_utility"]
            and other["E_customer_utility"] >= point["E_customer_utility"]
            and (
                other["E_supplier_utility"] > point["E_supplier_utility"]
                or other["E_customer_utility"] > point["E_customer_utility"]
            )
            for other in unique.values()
        )
    ]
    return len(pool), frontier


def make_contract(
    candidate: dict[str, Any],
    products: tuple[str, ...],
) -> Contract:
    quantities = candidate.get("quantity_schedule")
    if quantities is None:
        quantities = [candidate["quantities"]] * len(PRODUCTION_WEEKS)
    return Contract.from_mapping(
        {
            # F2's payment-deduction contingencies use an integer capital grid.
            # Keep the original F1 candidate unchanged in the output, but use
            # floored prices for this qualitative F2 evaluation contract.
            "product_prices": {
                product: math.floor(price)
                for product, price in zip(products, candidate["prices"])
            },
            "production_schedule": [
                {
                    "week": week,
                    "product_quantities": dict(zip(products, row)),
                }
                for week, row in zip(PRODUCTION_WEEKS, quantities)
            ],
            "payment_schedule": [
                {"week": week, "amount": amount}
                for week, amount in zip(PAYMENT_WEEKS, candidate["payments"])
            ],
        }
    )


def evaluate_contingencies(
    contract: Contract,
    environment: Environment,
    runtime: TransitionRuntime,
    grim_solver: RCCSolver,
) -> list[dict[str, Any]]:
    round_inputs = _round_inputs(contract, runtime)
    evaluations = []
    for spec in CONTINGENCY_SPECS:
        if spec.name == "grim_trigger":
            metrics = grim_solver.solve(contract, P_SAT_FLOOR)
        else:
            solver = _solver_for(spec, environment, contract)
            state = solver.solve_rounds_from_final_payment(
                round_inputs,
                contract.payment_schedule[-1],
                P_SAT_FLOOR,
            )
            metrics = _metrics(state, environment)
        evaluations.append({"contingency": spec.name, **metrics})
    return evaluations


def select_best(
    evaluations: list[dict[str, Any]], sweep: str, epsilon: float
) -> dict[str, Any] | None:
    constrained_role, optimized_role = roles(sweep)
    feasible = [
        evaluation
        for evaluation in evaluations
        if evaluation["P_sat"] + 1e-12 >= P_SAT_FLOOR
        and evaluation[f"E_{constrained_role}_utility"] + 1e-12 >= epsilon
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda point: point[f"E_{optimized_role}_utility"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument(
        "--sweep", choices=("max-customer", "max-supplier"), required=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    environment_root = DATA_ROOT / f"env-{args.env:02d}"
    pool_count, f1_frontier = load_role_frontier(environment_root, args.sweep)
    environment = Environment.from_json(
        REPOSITORY_ROOT / "envs" / str(args.env) / "environment.json"
    )
    runtime = TransitionRuntime(environment)
    grim_solver = RCCSolver(environment)
    constrained_role, optimized_role = roles(args.sweep)
    cache: dict[str, list[dict[str, Any]]] = {}
    output = environment_root / "f2-candidates" / f"{args.sweep}-f2.json"
    checkpoint = output.with_suffix(".checkpoint.json")
    points = []
    completed = 0

    if args.resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        if saved["environment_id"] != args.env or saved["sweep"] != args.sweep:
            raise ValueError("Checkpoint does not match this run")
        completed = saved["completed_f1_points"]
        points = saved["points"]

    def payload(status: str, completed_points: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "environment_id": args.env,
            "sweep": args.sweep,
            "source_point_count": pool_count,
            "f1_frontier_point_count": len(f1_frontier),
            "completed_f1_points": completed_points,
            "candidate_count": len(points),
            "points": points,
        }

    write_json(checkpoint, payload("running", completed))

    for index, point in enumerate(f1_frontier[completed:], completed + 1):
        key = json.dumps(point["candidate"], sort_keys=True)
        if key not in cache:
            contract = make_contract(
                point["candidate"],
                environment.supplier.products,
            )
            cache[key] = evaluate_contingencies(
                contract, environment, runtime, grim_solver
            )
        winner = select_best(cache[key], args.sweep, float(point["epsilon"]))
        if winner is not None:
            points.append(
                {
                    "f1_point": index - 1,
                    "f1_source": point["f1_source"],
                    "anchor_id": point.get("anchor_id"),
                    "epsilon": float(point["epsilon"]),
                    "sweep": args.sweep,
                    "epsilon_role": constrained_role,
                    "optimized_role": optimized_role,
                    "constraint_met": True,
                    "score": winner[f"E_{optimized_role}_utility"],
                    "candidate": point["candidate"],
                    **winner,
                }
            )
        print(f"[{index}/{len(f1_frontier)}] {args.sweep}", flush=True)
        write_json(checkpoint, payload("running", index))

    final = payload("complete", len(f1_frontier))
    write_json(output, final)
    write_json(checkpoint, final)
    print(f"Saved {len(points)} F2 candidates to {output}")


if __name__ == "__main__":
    main()
