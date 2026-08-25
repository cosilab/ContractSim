"""Environment-bound transition-kernel cache for repeated contract solves."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from contract import Contract
from environment import Environment
from environment.json_input import require_integer
from transitions import (
    ContractTransitionKernels,
    DeliveryKernel,
    ResourceTransitionKernel,
    SpoilageKernel,
)


class TransitionRuntime:
    """Compile environment transitions once and reuse them across contracts."""

    def __init__(self, environment: Environment) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")

        self.environment = environment
        self.delivery = DeliveryKernel(
            distribution=environment.probability_dist.delivery_distribution,
            max_inventory_per_resource=(
                environment.supplier.max_inventory_per_resource
            ),
            max_order_per_resource_per_round=(
                environment.supplier.max_order_per_resource_per_round
            ),
        )
        self.spoilage = SpoilageKernel(
            distribution=environment.probability_dist.spoilage_distribution,
            max_inventory_per_resource=(
                environment.supplier.max_inventory_per_resource
            ),
        )
        self._kernels_by_need: dict[int, ResourceTransitionKernel] = {}

    @property
    def kernels_by_need(self) -> Mapping[int, ResourceTransitionKernel]:
        """Expose the live cache through a read-only mapping."""
        return MappingProxyType(self._kernels_by_need)

    def resource_kernel(self, need: int) -> ResourceTransitionKernel:
        """Return the cached composed kernel for one required quantity."""
        normalized_need = require_integer(
            need,
            name="resource need",
            minimum=0,
        )
        if normalized_need not in self._kernels_by_need:
            self._kernels_by_need[normalized_need] = ResourceTransitionKernel(
                delivery=self.delivery,
                need=normalized_need,
                spoilage=self.spoilage,
            )
        return self._kernels_by_need[normalized_need]

    def for_contract(self, contract: Contract) -> ContractTransitionKernels:
        """Build a contract-specific round selection from shared kernels."""
        if not isinstance(contract, Contract):
            raise TypeError("contract must be a Contract")
        return ContractTransitionKernels(
            self.environment,
            contract,
            delivery=self.delivery,
            spoilage=self.spoilage,
            resource_kernel=self.resource_kernel,
        )
