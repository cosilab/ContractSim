"""Transition-kernel aggregate for one environment and contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from contract import Contract
from environment import Environment
from environment.json_input import require_integer

from .delivery import DeliveryKernel
from .resource import ResourceTransitionKernel
from .round import RoundTransitionKernels
from .spoilage import SpoilageKernel


@dataclass(frozen=True, eq=False, init=False)
class ContractTransitionKernels:
    """All reusable transition kernels required by one contract.

    The environment supplies declarative distribution specifications. Direct
    construction compiles delivery and spoilage arrays here; an environment-
    bound runtime may instead supply shared kernels and a resource-kernel
    resolver before contract-specific composition.
    """

    environment: Environment
    contract: Contract
    delivery: DeliveryKernel
    spoilage: SpoilageKernel
    kernels_by_need: Mapping[int, ResourceTransitionKernel]
    rounds: tuple[RoundTransitionKernels, ...]

    def __init__(
        self,
        environment: Environment,
        contract: Contract,
        *,
        delivery: DeliveryKernel | None = None,
        spoilage: SpoilageKernel | None = None,
        resource_kernel: Callable[[int], ResourceTransitionKernel] | None = None,
    ) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        if not isinstance(contract, Contract):
            raise TypeError("contract must be a Contract")

        supplier_products = set(environment.supplier.products)
        contract_products = set(contract.products)
        if contract_products != supplier_products:
            raise ValueError(
                "Contract products must match environment products: "
                f"contract={sorted(contract_products)}, "
                f"environment={sorted(supplier_products)}"
            )

        shared_values = (delivery, spoilage, resource_kernel)
        if any(value is not None for value in shared_values) and not all(
            value is not None for value in shared_values
        ):
            raise ValueError(
                "delivery, spoilage, and resource_kernel must be supplied together"
            )

        if delivery is None:
            delivery = DeliveryKernel(
                distribution=environment.probability_dist.delivery_distribution,
                max_inventory_per_resource=(
                    environment.supplier.max_inventory_per_resource
                ),
                max_order_per_resource_per_round=(
                    environment.supplier.max_order_per_resource_per_round
                ),
            )
            spoilage = SpoilageKernel(
                distribution=environment.probability_dist.spoilage_distribution,
                max_inventory_per_resource=(
                    environment.supplier.max_inventory_per_resource
                ),
            )
            local_cache: dict[int, ResourceTransitionKernel] = {}

            def resource_kernel(need: int) -> ResourceTransitionKernel:
                if need not in local_cache:
                    local_cache[need] = ResourceTransitionKernel(
                        delivery=delivery,
                        need=need,
                        spoilage=spoilage,
                    )
                return local_cache[need]
        elif not isinstance(delivery, DeliveryKernel):
            raise TypeError("delivery must be a DeliveryKernel")
        elif not isinstance(spoilage, SpoilageKernel):
            raise TypeError("spoilage must be a SpoilageKernel")
        elif not callable(resource_kernel):
            raise TypeError("resource_kernel must be callable")

        assert delivery is not None
        assert spoilage is not None
        assert resource_kernel is not None
        if (
            delivery.distribution
            != environment.probability_dist.delivery_distribution
            or delivery.max_inventory_per_resource
            != environment.supplier.max_inventory_per_resource
            or delivery.max_order_per_resource_per_round
            != environment.supplier.max_order_per_resource_per_round
        ):
            raise ValueError("delivery kernel must belong to the environment")
        if (
            spoilage.distribution
            != environment.probability_dist.spoilage_distribution
            or spoilage.max_inventory_per_resource
            != environment.supplier.max_inventory_per_resource
        ):
            raise ValueError("spoilage kernel must belong to the environment")
        resource_aliases = environment.supplier.resources
        kernels_by_need: dict[int, ResourceTransitionKernel] = {}
        rounds: list[RoundTransitionKernels] = []

        for production_week in contract.production_schedule:
            needs = {
                resource: sum(
                    int(production_week.product_quantities[product])
                    * int(environment.supplier.recipes[product].get(resource, 0))
                    for product in environment.supplier.products
                )
                for resource in resource_aliases
            }
            by_resource: dict[str, ResourceTransitionKernel] = {}
            for resource, need in needs.items():
                if need not in kernels_by_need:
                    kernel = resource_kernel(need)
                    if not isinstance(kernel, ResourceTransitionKernel):
                        raise TypeError(
                            "resource_kernel must return a "
                            "ResourceTransitionKernel"
                        )
                    if kernel.need != need:
                        raise ValueError(
                            "resource_kernel returned a kernel for the wrong need"
                        )
                    if kernel.delivery is not delivery or kernel.spoilage is not spoilage:
                        raise ValueError(
                            "resource kernel must use the supplied delivery and "
                            "spoilage kernels"
                        )
                    kernels_by_need[need] = kernel
                by_resource[resource] = kernels_by_need[need]

            rounds.append(
                RoundTransitionKernels(
                    week=production_week.week,
                    resource_needs=needs,
                    kernels_by_resource=by_resource,
                )
            )

        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(self, "spoilage", spoilage)
        object.__setattr__(
            self,
            "kernels_by_need",
            MappingProxyType(dict(sorted(kernels_by_need.items()))),
        )
        object.__setattr__(self, "rounds", tuple(rounds))

    @property
    def resource_aliases(self) -> tuple[str, ...]:
        return self.environment.supplier.resources

    def for_week(self, week: int) -> RoundTransitionKernels:
        """Return the transition-kernel selection for one production week."""
        normalized_week = require_integer(week, name="production week", minimum=1)
        for round_kernels in self.rounds:
            if round_kernels.week == normalized_week:
                return round_kernels
        raise KeyError(f"No production transition kernels for week {normalized_week}")
