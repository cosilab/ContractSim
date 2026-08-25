#!/usr/bin/env python3
"""Pareto-filter and plot generated F1 frontier candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_P_SAT_FLOOR = 0.95


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_point(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one serialized F1 point and retain plotting metadata."""
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


def _normalized_points(
    points: Iterable[Any],
) -> Iterable[dict[str, Any]]:
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
        # Match ContractBench's global F1 rule. The originating epsilon
        # constraint is subproblem metadata, not global frontier eligibility.
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_frontier(
    points: list[Mapping[str, Any]],
    output_directory: Path,
    *,
    title: str | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; install it or pass --no-figure"
        ) from exc

    supplier = [point["E_supplier_utility"] for point in points]
    customer = [point["E_customer_utility"] for point in points]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot(
        supplier,
        customer,
        color="#2563a6",
        linewidth=2,
        marker="o",
        markersize=4,
        label="F1 grim-trigger frontier",
    )
    axis.set_xlabel("Supplier expected utility")
    axis.set_ylabel("Customer expected utility")
    axis.set_title(title or "F1 Pareto frontier")
    axis.grid(alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_directory / "f1_frontier.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_directory / "f1_frontier.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.input]
    metadata, candidates = load_inputs(paths)
    frontier = pareto_envelope(candidates)
    if not frontier:
        raise ValueError("No feasible nondominated F1 points were found")

    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "frontier": "F1",
        **metadata,
        "sources": [str(path) for path in paths],
        "candidate_count": len(candidates),
        "frontier_point_count": len(frontier),
        "points": frontier,
    }
    write_json(output_directory / "f1_frontier.json", payload)
    write_csv(output_directory / "f1_frontier.csv", frontier)
    if not args.no_figure:
        plot_frontier(frontier, output_directory, title=args.title)
    print(
        f"Saved {len(frontier)} nondominated F1 points from "
        f"{len(candidates)} candidates to {output_directory}"
    )


if __name__ == "__main__":
    main()
