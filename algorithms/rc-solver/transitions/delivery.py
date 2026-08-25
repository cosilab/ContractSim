"""Delivery transition kernel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.json_input import require_integer
from environment.probability_dist import DeliveryDistributionSpec


def _minimum_received(order: int, rule: str) -> int:
    """Evaluate one of the delivery rules declared by the benchmark."""
    if rule == "ordered":
        return order
    if rule == "floor(3*ordered/4)":
        return 3 * order // 4
    if rule == "floor(ordered/2)":
        return order // 2
    raise ValueError(f"Unsupported delivery minimum rule: {rule}")


@dataclass(frozen=True, eq=False)
class DeliveryKernel:
    """Compile a delivery distribution into an inventory transition table."""

    distribution: DeliveryDistributionSpec
    max_inventory_per_resource: int
    max_order_per_resource_per_round: int
    table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, DeliveryDistributionSpec):
            raise TypeError("distribution must be a DeliveryDistributionSpec")
        max_inventory = require_integer(
            self.max_inventory_per_resource,
            name="max_inventory_per_resource",
            minimum=0,
        )
        max_order = require_integer(
            self.max_order_per_resource_per_round,
            name="max_order_per_resource_per_round",
            minimum=0,
        )

        if self.distribution.distribution != "discrete_uniform":
            raise ValueError("Only discrete_uniform delivery is currently supported")
        _minimum_received(0, self.distribution.minimum_received)
        if self.distribution.maximum_received != "ordered":
            raise ValueError("Only ordered maximum delivery is currently supported")
        if self.distribution.support_is_inclusive is not True:
            raise ValueError("Delivery support must be inclusive")

        number_of_inventory_states = max_inventory + 1
        number_of_order_states = max_order + 1
        table = np.zeros(
            (
                number_of_inventory_states,
                number_of_order_states,
                number_of_inventory_states,
            ),
            dtype=np.float64,
        )
        for inventory in range(number_of_inventory_states):
            for order in range(number_of_order_states):
                minimum_received = _minimum_received(
                    order,
                    self.distribution.minimum_received,
                )
                outcomes = range(minimum_received, order + 1)
                probability = 1.0 / len(outcomes)
                for received in outcomes:
                    resulting_inventory = min(
                        inventory + received,
                        max_inventory,
                    )
                    table[inventory, order, resulting_inventory] += probability
        table.setflags(write=False)

        object.__setattr__(self, "max_inventory_per_resource", max_inventory)
        object.__setattr__(
            self,
            "max_order_per_resource_per_round",
            max_order,
        )
        object.__setattr__(self, "table", table)

    @property
    def number_of_inventory_states(self) -> int:
        return int(self.table.shape[0])

    @property
    def number_of_order_states(self) -> int:
        return int(self.table.shape[1])
