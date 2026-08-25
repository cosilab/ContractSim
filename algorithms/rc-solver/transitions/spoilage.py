"""Spoilage transition kernel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.json_input import require_integer
from environment.probability_dist import SpoilageDistributionSpec


@dataclass(frozen=True, eq=False)
class SpoilageKernel:
    """Compile a spoilage distribution into an inventory transition table."""

    distribution: SpoilageDistributionSpec
    max_inventory_per_resource: int
    table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, SpoilageDistributionSpec):
            raise TypeError("distribution must be a SpoilageDistributionSpec")
        max_inventory = require_integer(
            self.max_inventory_per_resource,
            name="max_inventory_per_resource",
            minimum=0,
        )
        if self.distribution.independent_across_resources is not True:
            raise ValueError(
                "Only independent resource spoilage is currently supported"
            )

        number_of_inventory_states = max_inventory + 1
        table = np.zeros(
            (number_of_inventory_states, number_of_inventory_states),
            dtype=np.float64,
        )
        for leftover in range(number_of_inventory_states):
            for outcome in self.distribution.outcomes:
                table[leftover, max(leftover - outcome.units_lost, 0)] += (
                    outcome.probability
                )
        if (
            np.any(~np.isfinite(table))
            or np.any(table < 0)
            or not np.allclose(table.sum(axis=-1), 1.0, atol=1e-10)
        ):
            raise ValueError(
                "compiled spoilage kernel must be nonnegative and row-stochastic"
            )
        table.setflags(write=False)

        object.__setattr__(self, "max_inventory_per_resource", max_inventory)
        object.__setattr__(self, "table", table)

    @property
    def number_of_inventory_states(self) -> int:
        return int(self.table.shape[0])
