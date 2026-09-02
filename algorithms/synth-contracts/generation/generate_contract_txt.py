#!/usr/bin/env python3
"""Render contract.txt from a synthetic contract's JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BORDER = "=" * 50


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _item(count: int, name: str) -> str:
    return f"{count} {name if count == 1 else name + 's'}"


def _quantities(
    quantities: dict[str, Any],
    products: list[str],
    names: dict[str, str],
) -> str:
    return ", ".join(
        _item(int(quantities[product]), names[product]) for product in products
    )


def contract_text(
    terms: dict[str, Any],
    config: dict[str, Any],
    environment: dict[str, Any],
) -> str:
    base_game = environment["base_game"]
    products = sorted(terms["dish_prices"])
    names = {
        product: str(base_game["dishes"][product]["name"])
        for product in products
    }
    production = terms["production_schedule"]
    payments = terms["payment_schedule"]
    weeks = [int(row["week"]) for row in production + payments]
    contingencies = set(config["contingencies"])
    params = config.get("contingency_params", {})

    lines = [
        BORDER,
        "  AGREED CONTRACT",
        BORDER,
        "Parties: Customer and Caterer.",
        f"Term: Weeks {min(weeks)} through {max(weeks)}.",
        "Unit prices: "
        + "; ".join(
            f'{names[product]} ${int(terms["dish_prices"][product])}'
            for product in products
        )
        + ".",
        "Delivery obligations:",
        *[
            f'Week {int(row["week"])}: {_quantities(row, products, names)}.'
            for row in production
        ],
        "A shortfall is the amount by which a production week's delivery falls "
        "below its delivery obligation.",
        "Payments: "
        + "; ".join(
            f'Week {int(row["week"])} ${int(row["amount"])}' for row in payments
        )
        + ".",
    ]

    if "substitution" in contingencies:
        minimums = params["substitution"]["min_qty"]
        equivalents = {
            product: int(base_game["dishes"][product]["soup_equivalents"])
            for product in products
        }
        common_value = math.lcm(*equivalents.values())
        exchange = " = ".join(
            _item(common_value // equivalents[product], names[product])
            for product in products
        )
        lines.append(
            "Substitution: After delivering the non-substitutable minimum for "
            f"each dish ({_quantities(minimums, products, names)}), the Caterer "
            "may substitute quantities delivered above those minimums at "
            f"{exchange}, but only if the substitution fully satisfies that "
            "production week's delivery obligation. Partial fulfilment via "
            "substitution receives no credit; any resulting shortfall is "
            "calculated from actual physical deliveries and handled under the "
            "contract's other applicable clauses."
        )

    if "rollover" in contingencies:
        limit = int(params["rollover"]["max_deficit"])
        lines.append(
            f"Rollover: A shortfall by the Caterer of no more than {limit} units "
            "in each dish is added, dish by dish, to the next production week's "
            "delivery obligations to be cured. Carried shortfalls cannot be "
            "substituted. It is a violation if the shortfall in any dish exceeds "
            f"{limit} units, if the Caterer has a shortfall in two consecutive "
            "production weeks, or if a shortfall occurs in the final production "
            "week."
        )

    if "payment_deduction" in contingencies:
        lines.append(
            "Payment deduction: The Customer may reduce the next scheduled "
            "payment by the unit-price value of that production round's "
            "shortfall, to no less than $0; any deduction in excess of that "
            "payment is forfeited and does not carry to later payments; the "
            "deduction is final and creates no later repayment obligation. "
            "Absent a rollover clause, the Caterer need not make up the "
            "undelivered units. Two consecutive production weeks with a "
            "shortfall by the Caterer is a violation."
        )

    lines.extend(
        [
            "Termination: Except as modified by the clauses above, a violation "
            "by a party entitles the counterparty to cease all further "
            "performance immediately and permanently.",
            BORDER,
        ]
    )
    return "\n".join(lines) + "\n"


def write_contract_txt(contract_dir: Path, environment_root: Path) -> Path:
    """Render one generated contract directory in place."""
    terms = _read_json(contract_dir / "terms.json")
    config = _read_json(contract_dir / "config.json")
    environment = _read_json(
        environment_root / str(int(config["environment_id"])) / "environment.json"
    )
    output = contract_dir / "contract.txt"
    output.write_text(contract_text(terms, config, environment), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract_dir", type=Path)
    parser.add_argument(
        "--environment-root",
        type=Path,
        default=REPOSITORY_ROOT / "envs",
    )
    args = parser.parse_args()

    print(write_contract_txt(args.contract_dir, args.environment_root))


if __name__ == "__main__":
    main()
