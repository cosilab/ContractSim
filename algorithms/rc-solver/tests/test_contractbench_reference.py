from __future__ import annotations

import itertools
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from environment import Environment
from transitions.delivery import DeliveryKernel
from transitions.price import PriceKernel
from transitions.resource import ResourceTransitionKernel
from transitions.spoilage import SpoilageKernel


def reference_delivery(payload: dict, maximum_inventory: int, maximum_order: int):
    table = np.zeros(
        (maximum_inventory + 1, maximum_order + 1, maximum_inventory + 1),
        dtype=np.float64,
    )
    rule = payload["operating_conditions"]["delivery"]["minimum_received"]
    for inventory in range(maximum_inventory + 1):
        for order in range(maximum_order + 1):
            if rule == "ordered":
                low = order
            elif rule == "floor(3*ordered/4)":
                low = 3 * order // 4
            elif rule == "floor(ordered/2)":
                low = order // 2
            else:
                raise AssertionError(f"unexpected delivery rule: {rule}")
            probability = 1.0 / (order - low + 1)
            for received in range(low, order + 1):
                table[
                    inventory,
                    order,
                    min(inventory + received, maximum_inventory),
                ] += probability
    return table


def reference_spoilage(payload: dict, maximum_inventory: int):
    table = np.zeros(
        (maximum_inventory + 1, maximum_inventory + 1),
        dtype=np.float64,
    )
    outcomes = payload["operating_conditions"]["spoilage"]["outcomes"]
    for inventory in range(maximum_inventory + 1):
        for outcome in outcomes:
            remaining = max(inventory - int(outcome["units_lost"]), 0)
            table[inventory, remaining] += float(outcome["probability"])
    return table


def reference_prices(payload: dict):
    ingredient_outcomes = payload["operating_conditions"]["prices"][
        "ingredients"
    ]
    rows = [ingredient_outcomes[name]["outcomes"] for name in ("X", "Y", "Z")]
    joint = tuple(itertools.product(*rows))
    scenarios = np.asarray(
        [[int(outcome["value"]) for outcome in row] for row in joint],
        dtype=np.float32,
    )
    probabilities = np.asarray(
        [math.prod(float(outcome["probability"]) for outcome in row) for row in joint],
        dtype=np.float64,
    )
    return scenarios, probabilities


def reference_resource(delivery, spoilage, need: int):
    number_of_states, number_of_orders, _ = delivery.shape
    failure = number_of_states
    table = np.zeros(
        (number_of_states, number_of_orders, number_of_states + 1),
        dtype=np.float64,
    )
    for inventory in range(number_of_states):
        for order in range(number_of_orders):
            for effective, received_probability in enumerate(
                delivery[inventory, order]
            ):
                if received_probability == 0.0:
                    continue
                if effective < need:
                    table[inventory, order, failure] += received_probability
                    continue
                leftover = effective - need
                for next_inventory, spoilage_probability in enumerate(
                    spoilage[leftover]
                ):
                    if spoilage_probability != 0.0:
                        table[inventory, order, next_inventory] += (
                            received_probability * spoilage_probability
                        )
    return table


class ContractBenchKernelReferenceTests(unittest.TestCase):
    def test_raw_kernels_match_contractbench_order_for_every_environment(self):
        for environment_id in range(1, 7):
            with self.subTest(environment_id=environment_id):
                path = (
                    PROJECT_ROOT
                    / "envs"
                    / str(environment_id)
                    / "environment.json"
                )
                payload = json.loads(path.read_text())
                environment = Environment(payload)
                supplier = environment.supplier
                delivery = DeliveryKernel(
                    environment.probability_dist.delivery_distribution,
                    supplier.max_inventory_per_resource,
                    supplier.max_order_per_resource_per_round,
                )
                spoilage = SpoilageKernel(
                    environment.probability_dist.spoilage_distribution,
                    supplier.max_inventory_per_resource,
                )
                prices = PriceKernel(
                    environment.probability_dist.price_distribution
                )
                expected_prices, expected_probabilities = reference_prices(payload)

                np.testing.assert_array_equal(
                    delivery.table,
                    reference_delivery(
                        payload,
                        supplier.max_inventory_per_resource,
                        supplier.max_order_per_resource_per_round,
                    ),
                )
                np.testing.assert_array_equal(
                    spoilage.table,
                    reference_spoilage(
                        payload,
                        supplier.max_inventory_per_resource,
                    ),
                )
                np.testing.assert_array_equal(prices.scenarios, expected_prices)
                np.testing.assert_array_equal(
                    prices.probabilities,
                    expected_probabilities,
                )

    def test_composed_need_tables_match_the_reference_loop(self):
        for environment_id in (1, 3, 5):
            path = (
                PROJECT_ROOT
                / "envs"
                / str(environment_id)
                / "environment.json"
            )
            environment = Environment.from_json(path)
            supplier = environment.supplier
            delivery = DeliveryKernel(
                environment.probability_dist.delivery_distribution,
                supplier.max_inventory_per_resource,
                supplier.max_order_per_resource_per_round,
            )
            spoilage = SpoilageKernel(
                environment.probability_dist.spoilage_distribution,
                supplier.max_inventory_per_resource,
            )
            for need in range(supplier.max_order_per_resource_per_round + 1):
                with self.subTest(environment_id=environment_id, need=need):
                    actual = ResourceTransitionKernel(
                        delivery,
                        need,
                        spoilage,
                    )
                    np.testing.assert_array_equal(
                        actual.table,
                        reference_resource(delivery.table, spoilage.table, need),
                    )


if __name__ == "__main__":
    unittest.main()
