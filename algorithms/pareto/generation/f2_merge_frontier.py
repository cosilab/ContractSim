#!/usr/bin/env python3
"""Merge the two completed F2 role sweeps into one Pareto frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def merge_environment(data_root: Path, environment_id: int) -> Path:
    environment_root = data_root / f"env-{environment_id:02d}"
    sources = [
        environment_root / "f2-candidates" / "max-customer-f2.json",
        environment_root / "f2-candidates" / "max-supplier-f2.json",
    ]
    payloads = [json.loads(path.read_text()) for path in sources]
    if any(payload.get("status") != "complete" for payload in payloads):
        raise RuntimeError(f"F2 inputs are not complete for environment {environment_id}")

    points = [point for payload in payloads for point in payload["points"]]
    unique = {
        (
            round(float(point["E_supplier_utility"]), 8),
            round(float(point["E_customer_utility"]), 8),
        ): point
        for point in points
    }

    frontier = []
    best_customer = float("-inf")
    for point in sorted(
        unique.values(),
        key=lambda row: (
            float(row["E_supplier_utility"]),
            float(row["E_customer_utility"]),
        ),
        reverse=True,
    ):
        if float(point["E_customer_utility"]) > best_customer:
            frontier.append(point)
            best_customer = float(point["E_customer_utility"])
    frontier.reverse()

    output = environment_root / "f2-merged" / "f2_frontier.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frontier": "F2",
                "environment_id": environment_id,
                "sources": [str(path.resolve()) for path in sources],
                "candidate_count": len(points),
                "frontier_point_count": len(frontier),
                "points": frontier,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--env", type=int, nargs="+", default=[3, 4, 5, 6])
    args = parser.parse_args()

    for environment_id in args.env:
        output = merge_environment(args.data_root.resolve(), environment_id)
        print(f"Saved F2 frontier for env-{environment_id:02d} to {output}")


if __name__ == "__main__":
    main()
