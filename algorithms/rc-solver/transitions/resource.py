"""Composed transition kernel for one required resource quantity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.json_input import require_integer

from .delivery import DeliveryKernel
from .spoilage import SpoilageKernel


@dataclass(frozen=True, eq=False)
class ResourceTransitionKernel:
    """Delivery, consumption, and spoilage composed for one resource need."""

    delivery: DeliveryKernel
    need: int
    spoilage: SpoilageKernel
    table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.delivery, DeliveryKernel):
            raise TypeError("delivery must be a DeliveryKernel")
        if not isinstance(self.spoilage, SpoilageKernel):
            raise TypeError("spoilage must be a SpoilageKernel")
        need = require_integer(self.need, name="resource need", minimum=0)

        number_of_states = self.delivery.number_of_inventory_states
        if self.spoilage.number_of_inventory_states != number_of_states:
            raise ValueError("spoilage inventory states must match the delivery kernel")

        # Accumulate delivery, deterministic consumption, then spoilage.
        failure_state = number_of_states
        table = np.zeros(
            (
                number_of_states,
                self.delivery.number_of_order_states,
                number_of_states + 1,
            ),
            dtype=np.float64,
        )
        for inventory in range(number_of_states):
            for order in range(self.delivery.number_of_order_states):
                for effective_inventory, delivery_probability in enumerate(
                    self.delivery.table[inventory, order]
                ):
                    if delivery_probability == 0.0:
                        continue
                    if effective_inventory < need:
                        table[inventory, order, failure_state] += delivery_probability
                        continue
                    leftover = effective_inventory - need
                    for next_inventory, spoilage_probability in enumerate(
                        self.spoilage.table[leftover]
                    ):
                        if spoilage_probability == 0.0:
                            continue
                        table[inventory, order, next_inventory] += (
                            delivery_probability * spoilage_probability
                        )
        if not np.allclose(table.sum(axis=-1), 1.0, atol=1e-10):
            raise ValueError("composed resource transition must be row-stochastic")
        table.setflags(write=False)
        object.__setattr__(self, "need", need)
        object.__setattr__(self, "table", table)

    @property
    def failure_state(self) -> int:
        return self.delivery.number_of_inventory_states

    @property
    def successful_table(self) -> np.ndarray:
        """Return the current solver's sub-stochastic, success-only view."""
        return self.table[..., : self.failure_state]
