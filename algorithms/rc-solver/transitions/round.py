"""Transition-kernel selection for one contract production round."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from environment.json_input import require_integer

from .resource import ResourceTransitionKernel


@dataclass(frozen=True, eq=False)
class RoundTransitionKernels:
    """Resource needs and composed kernels for one production week."""

    week: int
    resource_needs: Mapping[str, int]
    kernels_by_resource: Mapping[str, ResourceTransitionKernel]

    def __post_init__(self) -> None:
        week = require_integer(self.week, name="production week", minimum=1)
        needs: dict[str, int] = {}
        for resource, raw_need in self.resource_needs.items():
            if not isinstance(resource, str) or not resource:
                raise ValueError("resource aliases must be non-empty strings")
            needs[resource] = require_integer(
                raw_need,
                name=f"resource need for {resource}",
                minimum=0,
            )

        kernels = dict(self.kernels_by_resource)
        if set(kernels) != set(needs):
            raise ValueError("round kernel resources must match round resource needs")
        for resource, kernel in kernels.items():
            if not isinstance(kernel, ResourceTransitionKernel):
                raise TypeError(
                    f"kernel for resource {resource!r} must be a "
                    "ResourceTransitionKernel"
                )
            if kernel.need != needs[resource]:
                raise ValueError(
                    f"kernel need for resource {resource!r} does not match the round"
                )

        object.__setattr__(self, "week", week)
        object.__setattr__(self, "resource_needs", MappingProxyType(needs))
        object.__setattr__(self, "kernels_by_resource", MappingProxyType(kernels))
