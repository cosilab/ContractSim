"""Compilation of declarative resource-price distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod

import numpy as np

from environment.probability_dist import PriceDistributionSpec


@dataclass(frozen=True, eq=False)
class PriceKernel:
    """Compile independent resource prices into joint scenarios and weights."""

    distribution: PriceDistributionSpec
    scenarios: np.ndarray = field(init=False, repr=False)
    probabilities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, PriceDistributionSpec):
            raise TypeError("distribution must be a PriceDistributionSpec")
        if self.distribution.independent_across_resources is not True:
            raise ValueError("Only independent resource prices are currently supported")

        per_resource = [
            self.distribution.outcomes_by_resource[resource]
            for resource in self.distribution.resource_names
        ]
        scenarios: list[list[float]] = []
        probabilities: list[float] = []
        for joint_outcome in product(*per_resource):
            scenarios.append([outcome.value for outcome in joint_outcome])
            probabilities.append(
                prod(float(outcome.probability) for outcome in joint_outcome)
            )

        scenario_array = np.asarray(scenarios, dtype=np.float32)
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if (
            np.any(~np.isfinite(scenario_array))
            or np.any(scenario_array < 0)
        ):
            raise ValueError("compiled price scenarios must be finite and nonnegative")
        if (
            np.any(~np.isfinite(probability_array))
            or np.any(probability_array < 0)
            or not np.isclose(probability_array.sum(), 1.0, atol=1e-10)
        ):
            raise ValueError(
                "compiled price probabilities must be nonnegative and sum to one"
            )
        scenario_array.setflags(write=False)
        probability_array.setflags(write=False)
        object.__setattr__(self, "scenarios", scenario_array)
        object.__setattr__(self, "probabilities", probability_array)

    @property
    def resource_names(self) -> tuple[str, ...]:
        return self.distribution.resource_names

    def as_solver_mapping(self) -> dict[str, np.ndarray]:
        """Return the price-array names used by the current solver."""
        return {
            "PRICE_RANGES": self.scenarios,
            "PRICE_PROB": self.probabilities,
        }
