"""Aggregate environment model for the RC solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .customer import CustomerSpec
from .json_input import (
    load_json_object,
    require_integer,
    require_json_mapping,
    require_nonempty_string,
)
from .probability_dist import ProbabilityDist
from .supplier import SupplierSpec


def _axes(value: Any) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("axes must be a mapping")
    normalized: dict[str, str] = {}
    for key, selection in value.items():
        axis_name = require_nonempty_string(key, name="axis name")
        normalized[axis_name] = require_nonempty_string(
            selection,
            name=f"axis {key!r} selection",
        )
    return MappingProxyType(normalized)


def _supported_contingencies(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("supported_contingencies must be a list")
    normalized: list[str] = []
    for item in value:
        name = require_nonempty_string(item, name="supported contingency")
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


@dataclass(frozen=True, init=False)
class Environment:
    """Complete immutable environment built only from JSON-shaped input."""

    schema_version: int
    environment_id: int
    environment_name: str
    customer: CustomerSpec
    supplier: SupplierSpec
    probability_dist: ProbabilityDist
    axes: Mapping[str, str]
    supported_contingencies: tuple[str, ...] = ()
    random_seed: int | None = None

    def __init__(self, source: Mapping[str, Any]) -> None:
        values = self._parse_mapping(source)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        schema_version = require_integer(
            self.schema_version,
            name="schema_version",
            minimum=1,
        )
        environment_id = require_integer(
            self.environment_id,
            name="environment_id",
            minimum=1,
        )
        environment_name = require_nonempty_string(
            self.environment_name,
            name="environment_name",
        )
        if not isinstance(self.customer, CustomerSpec):
            raise TypeError("customer must be a CustomerSpec")
        if not isinstance(self.supplier, SupplierSpec):
            raise TypeError("supplier must be a SupplierSpec")
        if not isinstance(self.probability_dist, ProbabilityDist):
            raise TypeError("probability_dist must be a ProbabilityDist")

        axes = _axes(self.axes)
        supported_contingencies = _supported_contingencies(
            list(self.supported_contingencies)
        )
        random_seed = (
            None
            if self.random_seed is None
            else require_integer(self.random_seed, name="random_seed", minimum=0)
        )

        customer_products = set(self.customer.private_values)
        supplier_products = set(self.supplier.products)
        if customer_products != supplier_products:
            raise ValueError(
                "Customer products must match supplier recipe products: "
                f"customer={sorted(customer_products)}, "
                f"supplier={sorted(supplier_products)}"
            )

        supplier_resources = tuple(self.supplier.resources)
        if self.probability_dist.resource_names != supplier_resources:
            raise ValueError(
                "Probability-distribution resources must match supplier resources"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(self, "environment_name", environment_name)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(
            self,
            "supported_contingencies",
            supported_contingencies,
        )
        object.__setattr__(self, "random_seed", random_seed)

    @staticmethod
    def _parse_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
        source = require_json_mapping(source, name="Environment input")

        try:
            schema_version = source["schema_version"]
            environment_id = source["environment_id"]
            environment_name = source["environment_name"]
        except KeyError as exc:
            raise ValueError(f"Missing environment field: {exc.args[0]}") from exc

        customer = CustomerSpec(source)
        supplier = SupplierSpec(source)
        probability_dist = ProbabilityDist(source)

        base_game = source.get("base_game")
        if not isinstance(base_game, Mapping):
            raise ValueError("environment.base_game must be a mapping")
        contingency_config = base_game.get("contingencies", {})
        if not isinstance(contingency_config, Mapping):
            raise ValueError("environment.base_game.contingencies must be a mapping")

        capital = source.get("caterer_capital")
        if isinstance(capital, Mapping) and "customer_budget" in capital:
            if float(capital["customer_budget"]) != customer.starting_budget:
                raise ValueError(
                    "caterer_capital.customer_budget must match customer starting_budget"
                )

        kernel_metadata = source.get("probability_kernel")
        if isinstance(kernel_metadata, Mapping):
            declared_kernel_id = kernel_metadata.get("kernel_id")
            if (
                declared_kernel_id is not None
                and declared_kernel_id != probability_dist.kernel_id
            ):
                raise ValueError(
                    "Declared probability distribution does not match operating conditions"
                )
            shared_ids = kernel_metadata.get("shared_by_environment_ids")
            if isinstance(shared_ids, list) and int(environment_id) not in {
                int(item) for item in shared_ids
            }:
                raise ValueError(
                    "Environment ID is not listed for the declared probability kernel"
                )

        return {
            "schema_version": schema_version,
            "environment_id": environment_id,
            "environment_name": environment_name,
            "customer": customer,
            "supplier": supplier,
            "probability_dist": probability_dist,
            "axes": source.get("axes", {}),
            "supported_contingencies": _supported_contingencies(
                contingency_config.get("supported")
            ),
            "random_seed": source.get("random_seed"),
        }

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "Environment":
        """Build from an already parsed JSON object."""
        return cls(source)

    @classmethod
    def from_json(cls, path: str | Path) -> "Environment":
        """Load an environment JSON file."""
        return cls(load_json_object(path, name="Environment JSON"))
