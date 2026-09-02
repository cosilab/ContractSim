#!/usr/bin/env python3
"""Merge base and refined role sweeps into one final F1 frontier."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_P_SAT_FLOOR = 0.95
PARETO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PARETO_ROOT / "data"


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_point(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one serialized F1 point and retain frontier metadata."""
    return {
        "epsilon": _finite_number(payload["epsilon"], name="epsilon"),
        "sweep": str(payload["sweep"]),
        "P_sat": _finite_number(payload["P_sat"], name="P_sat"),
        "E_supplier_utility": _finite_number(
            payload["E_supplier_utility"], name="E_supplier_utility"
        ),
        "E_customer_utility": _finite_number(
            payload["E_customer_utility"], name="E_customer_utility"
        ),
        "constraint_met": bool(payload["constraint_met"]),
        "candidate": payload.get("candidate"),
    }


def _normalized_points(points: Iterable[Any]) -> Iterable[dict[str, Any]]:
    """Yield valid numeric rows, skipping malformed older records."""
    for point in points:
        if not isinstance(point, Mapping):
            continue
        try:
            yield normalize_point(point)
        except (KeyError, TypeError, ValueError):
            continue


def pareto_envelope(points: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return unique, globally eligible, point-Pareto-maximal F1 candidates."""
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for point in _normalized_points(points):
        # The originating epsilon constraint is subproblem metadata, not a
        # global eligibility condition.
        if point["P_sat"] + 1e-12 < DEFAULT_P_SAT_FLOOR:
            continue
        pair = (
            round(point["E_supplier_utility"], 8),
            round(point["E_customer_utility"], 8),
        )
        unique.setdefault(pair, point)

    best_customer = -math.inf
    envelope: list[dict[str, Any]] = []
    for point in sorted(
        unique.values(),
        key=lambda row: (row["E_supplier_utility"], row["E_customer_utility"]),
        reverse=True,
    ):
        if point["E_customer_utility"] > best_customer:
            envelope.append(point)
            best_customer = point["E_customer_utility"]
    envelope.reverse()
    return envelope


def load_f1_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {}, list(_normalized_points(payload))
    if not isinstance(payload, dict):
        raise ValueError(f"F1 input must contain an object or list: {path}")
    raw_points = payload.get("points", payload.get("completed_points"))
    if not isinstance(raw_points, list):
        raise ValueError(f"F1 input has no points list: {path}")
    metadata = {
        key: payload[key]
        for key in (
            "environment_id",
            "environment_name",
            "environment_axes",
            "environment_fingerprint",
        )
        if key in payload
    }
    return metadata, list(_normalized_points(raw_points))


def load_inputs(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for path in paths:
        metadata, rows = load_f1_file(path)
        metadata_rows.append(metadata)
        points.extend(rows)

    environment_ids = {
        int(metadata["environment_id"])
        for metadata in metadata_rows
        if "environment_id" in metadata
    }
    fingerprints = {
        str(metadata["environment_fingerprint"])
        for metadata in metadata_rows
        if "environment_fingerprint" in metadata
    }
    if len(environment_ids) > 1:
        raise ValueError(f"F1 inputs mix environment IDs: {sorted(environment_ids)}")
    if len(fingerprints) > 1:
        raise ValueError("F1 inputs were generated from different environment snapshots")
    metadata = next((row for row in metadata_rows if row), {})
    return metadata, points


def default_sources(data_root: Path, environment_id: int) -> list[Path]:
    environment_root = data_root / f"env-{environment_id:02d}"
    return [
        environment_root / "f1-base" / "max-customer-f1.json",
        environment_root / "f1-base" / "max-supplier-f1.json",
        environment_root / "f1-refinement" / "max-customer-f1.json",
        environment_root / "f1-refinement" / "max-supplier-f1.json",
    ]


def build_frontier_payload(paths: list[Path]) -> dict[str, Any]:
    resolved_paths = [path.resolve() for path in paths]
    metadata, candidates = load_inputs(resolved_paths)
    if "environment_id" not in metadata:
        raise ValueError("F1 inputs do not identify an environment")
    frontier = pareto_envelope(candidates)
    if not frontier:
        raise ValueError("No eligible nondominated F1 points were found")
    return {
        "schema_version": 1,
        "frontier": "F1",
        **metadata,
        "sources": [str(path) for path in resolved_paths],
        "candidate_count": len(candidates),
        "frontier_point_count": len(frontier),
        "points": frontier,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, points: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "E_supplier_utility",
        "E_customer_utility",
        "P_sat",
        "sweep",
        "epsilon",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)


def write_frontier_outputs(output_directory: Path, payload: Mapping[str, Any]) -> None:
    write_json(output_directory / "f1_frontier.json", payload)
    write_csv(output_directory / "f1_frontier.csv", list(payload["points"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir
        else data_root / f"env-{args.env:02d}" / "f1-merged"
    )
    payload = build_frontier_payload(default_sources(data_root, args.env))
    if int(payload["environment_id"]) != args.env:
        raise ValueError(
            f"Merged inputs identify environment {payload['environment_id']}, "
            f"not requested environment {args.env}"
        )
    write_frontier_outputs(output_directory, payload)
    print(
        f"Saved {payload['frontier_point_count']} nondominated F1 points from "
        f"{payload['candidate_count']} base/refined candidates to {output_directory}"
    )


if __name__ == "__main__":
    main()
