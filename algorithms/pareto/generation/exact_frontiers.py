#!/usr/bin/env python3
"""Generate exact full-performance F1/F2 frontiers for benign envs."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path


PARETO_DIR = Path(__file__).resolve().parents[1]
ROOT = PARETO_DIR.parents[1]
ROUNDS = 5
PRODUCTS = ("A", "B", "C")
DEFAULT_OUTPUT_ROOT = PARETO_DIR / "data"


def load_environment(environment_id: int) -> dict:
    path = ROOT / "envs" / str(environment_id) / "environment.json"
    environment = json.loads(path.read_text())
    if environment["environment_id"] != environment_id:
        raise ValueError(f"Environment ID mismatch in {path}")
    return environment


def one_outcome(outcomes: list[dict], key: str) -> int:
    if len(outcomes) != 1 or outcomes[0]["probability"] != 1.0:
        raise ValueError("This script requires deterministic environments")
    return int(outcomes[0][key])


def parameters(environment: dict) -> dict:
    operating = environment["operating_conditions"]
    prices = {
        ingredient: one_outcome(spec["outcomes"], "value")
        for ingredient, spec in operating["prices"]["ingredients"].items()
    }
    recipes = environment["base_game"]["recipes"]
    costs = {
        product: sum(amount * prices[ingredient] for ingredient, amount in recipes[product].items())
        for product in PRODUCTS
    }
    return {
        "costs": costs,
        "values": environment["customer_preference"]["private_values"],
        "minimum": environment["base_game"]["customer"][
            "minimum_dishes_per_production_round"
        ],
        "cap": environment["base_game"]["inventory"]["max_units_per_ingredient"],
        "budget": environment["base_game"]["customer"]["starting_budget"],
        "recipes": recipes,
    }


def legal_bundles(params: dict) -> list[tuple[int, int, int]]:
    bundles = []
    for quantities in product(
        *(range(params["minimum"][product], params["cap"] + 1) for product in PRODUCTS)
    ):
        if all(
            sum(quantities[index] * params["recipes"][product].get(ingredient, 0)
                for index, product in enumerate(PRODUCTS))
            <= params["cap"]
            for ingredient in ("X", "Y", "Z")
        ):
            bundles.append(tuple(quantities))
    return bundles


def quantity_layers(
    bundles: list[tuple[int, int, int]],
) -> list[set[tuple[int, int, int]]]:
    layers = [{(0, 0, 0)}]
    for _ in range(ROUNDS):
        layers.append({
            tuple(previous[index] + bundle[index] for index in range(3))
            for previous in layers[-1]
            for bundle in bundles
        })
    return layers


def example_schedule(
    total: tuple[int, int, int],
    bundles: list[tuple[int, int, int]],
    layers: list[set[tuple[int, int, int]]],
) -> list[dict[str, int]]:
    remaining = total
    schedule = []
    for layer in range(ROUNDS, 0, -1):
        for bundle in bundles:
            previous = tuple(remaining[index] - bundle[index] for index in range(3))
            if previous in layers[layer - 1]:
                schedule.append(dict(zip(PRODUCTS, bundle)))
                remaining = previous
                break
    return list(reversed(schedule))


def exact_frontier(environment_id: int, payment_floor: int) -> list[dict]:
    environment = load_environment(environment_id)
    params = parameters(environment)
    bundles = legal_bundles(params)
    layers = quantity_layers(bundles)
    best_by_supplier_utility = {}

    price_ranges = [
        range(params["costs"][product], params["values"][product] + 1)
        for product in PRODUCTS
    ]
    for quantities in layers[-1]:
        cost = sum(
            quantities[index] * params["costs"][product]
            for index, product in enumerate(PRODUCTS)
        )
        value = sum(
            quantities[index] * params["values"][product]
            for index, product in enumerate(PRODUCTS)
        )
        for prices in product(*price_ranges):
            payment = sum(quantities[index] * prices[index] for index in range(3))
            if not payment_floor <= payment <= params["budget"]:
                continue
            supplier_utility = payment - cost
            customer_utility = params["budget"] + value - payment
            point = {
                "environment_id": environment_id,
                "epsilon": 0,
                "sweep": "exact",
                "constraint_met": True,
                "P_sat": 1.0,
                "E_supplier_utility": supplier_utility,
                "E_customer_utility": customer_utility,
                "total_payment": payment,
                "total_production_cost": cost,
                "total_customer_value": value,
                "quantity_A": quantities[0],
                "quantity_B": quantities[1],
                "quantity_C": quantities[2],
                "price_A": prices[0],
                "price_B": prices[1],
                "price_C": prices[2],
                "payments": [payment, 0, 0, 0, 0, 0],
            }
            previous = best_by_supplier_utility.get(supplier_utility)
            if previous is None or customer_utility > previous["E_customer_utility"]:
                best_by_supplier_utility[supplier_utility] = point

    frontier = []
    best_customer_utility = float("-inf")
    for supplier_utility in sorted(best_by_supplier_utility, reverse=True):
        point = best_by_supplier_utility[supplier_utility]
        if point["E_customer_utility"] > best_customer_utility:
            frontier.append(point)
            best_customer_utility = point["E_customer_utility"]
    frontier = list(reversed(frontier))
    for point in frontier:
        quantities = tuple(point[f"quantity_{product}"] for product in PRODUCTS)
        point["contract_qty_schedule"] = example_schedule(quantities, bundles, layers)
    return frontier


def write_outputs(output_dir: Path, environment: dict, points: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frontier_name in ("F1", "F2"):
        payload = {
            "schema_version": 1,
            "frontier": frontier_name,
            "status": "complete",
            "environment_id": environment["environment_id"],
            "environment_name": environment["environment_name"],
            "method": "exact deterministic enumeration",
            "points": points,
        }
        (output_dir / f"{frontier_name.lower()}_frontier.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    fields = (
        "E_supplier_utility",
        "E_customer_utility",
        "P_sat",
        "total_payment",
        "quantity_A",
        "quantity_B",
        "quantity_C",
        "price_A",
        "price_B",
        "price_C",
    )
    with (output_dir / "frontier.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)

    try:
        import matplotlib
    except ModuleNotFoundError:
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [point["E_customer_utility"] for point in points]
    y = [point["E_supplier_utility"] for point in points]
    plt.plot(x, y, "o-", label="F1")
    plt.plot(x, y, "s--", label="F2")
    plt.xlabel("Customer utility")
    plt.ylabel("Supplier utility")
    plt.title(environment["environment_name"])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--payment-floor", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    for environment_id in args.env:
        environment = load_environment(environment_id)
        points = exact_frontier(environment_id, args.payment_floor)
        output_dir = args.output_root / f"env-{environment_id:02d}"
        write_outputs(output_dir, environment, points)
        print(f"env-{environment_id:02d}: wrote {len(points)} frontier points to {output_dir}")


if __name__ == "__main__":
    main()
