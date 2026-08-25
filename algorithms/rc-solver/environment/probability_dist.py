"""Declarative probability-distribution specifications for an environment.

This module deliberately does not build solver transition tables. It parses
and validates the stochastic rules declared by environment JSON. Compiled
delivery, spoilage, and price arrays are owned by the corresponding classes in
the ``transitions`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .json_input import (
    load_json_object,
    require_finite_number,
    require_integer,
    require_json_mapping,
    require_mapping,
    require_nonempty_string,
    require_probability,
    validate_probability_sum,
)
from .supplier import SupplierSpec


@dataclass(frozen=True)
class DeliveryDistributionSpec:
    """Declarative delivery distribution from the environment."""

    distribution: str
    minimum_received: str
    maximum_received: str
    support_is_inclusive: bool


@dataclass(frozen=True)
class SpoilageOutcome:
    """One possible number of spoiled units and its probability."""

    units_lost: int
    probability: float


@dataclass(frozen=True)
class SpoilageDistributionSpec:
    """Declarative spoilage distribution from the environment."""

    independent_across_resources: bool
    outcomes: tuple[SpoilageOutcome, ...]


@dataclass(frozen=True)
class PriceOutcome:
    """One possible resource price and its probability."""

    value: float
    probability: float


@dataclass(frozen=True)
class PriceDistributionSpec:
    """Declarative per-resource price distributions."""

    independent_across_resources: bool
    outcomes_by_resource: Mapping[str, tuple[PriceOutcome, ...]]

    @property
    def resource_names(self) -> tuple[str, ...]:
        return tuple(self.outcomes_by_resource)


def _parse_delivery(value: Mapping[str, Any]) -> DeliveryDistributionSpec:
    support_is_inclusive = value.get("support_is_inclusive")
    if not isinstance(support_is_inclusive, bool):
        raise ValueError("delivery support_is_inclusive must be a boolean")
    return DeliveryDistributionSpec(
        distribution=require_nonempty_string(
            value.get("distribution"),
            name="delivery distribution",
        ),
        minimum_received=require_nonempty_string(
            value.get("minimum_received"),
            name="delivery minimum_received",
        ),
        maximum_received=require_nonempty_string(
            value.get("maximum_received"),
            name="delivery maximum_received",
        ),
        support_is_inclusive=support_is_inclusive,
    )


def _parse_spoilage(value: Mapping[str, Any]) -> SpoilageDistributionSpec:
    independent = value.get("independent_across_ingredients")
    if not isinstance(independent, bool):
        raise ValueError(
            "spoilage independent_across_ingredients must be a boolean"
        )

    raw_outcomes = value.get("outcomes")
    if not isinstance(raw_outcomes, list) or not raw_outcomes:
        raise ValueError("operating_conditions.spoilage.outcomes must be a list")

    outcomes: list[SpoilageOutcome] = []
    for index, raw_outcome in enumerate(raw_outcomes):
        if not isinstance(raw_outcome, Mapping):
            raise ValueError(f"spoilage outcome {index} must be a mapping")
        outcomes.append(
            SpoilageOutcome(
                units_lost=require_integer(
                    raw_outcome.get("units_lost"),
                    name=f"spoilage outcome {index} units_lost",
                    minimum=0,
                ),
                probability=require_probability(
                    raw_outcome.get("probability"),
                    name=f"spoilage outcome {index} probability",
                ),
            )
        )
    validate_probability_sum(
        [outcome.probability for outcome in outcomes],
        name="spoilage outcome",
    )
    return SpoilageDistributionSpec(
        independent_across_resources=independent,
        outcomes=tuple(outcomes),
    )


def _parse_prices(
    value: Mapping[str, Any],
    *,
    expected_resources: tuple[str, ...],
) -> PriceDistributionSpec:
    independent = value.get("independent_across_ingredients")
    if not isinstance(independent, bool):
        raise ValueError("prices independent_across_ingredients must be a boolean")

    resources = require_mapping(
        value,
        "ingredients",
        path="environment.operating_conditions.prices",
    )
    if set(resources) != set(expected_resources):
        raise ValueError(
            "Price resources must match recipe resources: "
            f"expected {sorted(expected_resources)}, got {sorted(resources)}"
        )

    normalized: dict[str, tuple[PriceOutcome, ...]] = {}
    for resource in expected_resources:
        resource_spec = resources[resource]
        if not isinstance(resource_spec, Mapping):
            raise ValueError(f"Price specification for {resource} must be a mapping")
        raw_outcomes = resource_spec.get("outcomes")
        if not isinstance(raw_outcomes, list) or not raw_outcomes:
            raise ValueError(f"Price outcomes for {resource} must be a list")

        outcomes: list[PriceOutcome] = []
        for index, raw_outcome in enumerate(raw_outcomes):
            if not isinstance(raw_outcome, Mapping):
                raise ValueError(
                    f"Price outcome {index} for {resource} must be a mapping"
                )
            outcomes.append(
                PriceOutcome(
                    value=require_finite_number(
                        raw_outcome.get("value"),
                        name=f"price outcome {index} for {resource} value",
                        minimum=0,
                    ),
                    probability=require_probability(
                        raw_outcome.get("probability"),
                        name=f"price outcome {index} for {resource}",
                    ),
                )
            )
        validate_probability_sum(
            [outcome.probability for outcome in outcomes],
            name=f"price outcomes for {resource}",
        )
        normalized[resource] = tuple(outcomes)

    return PriceDistributionSpec(
        independent_across_resources=independent,
        outcomes_by_resource=MappingProxyType(normalized),
    )


@dataclass(frozen=True, init=False)
class ProbabilityDist:
    """Validated declarative stochastic rules built from environment JSON."""

    delivery_distribution: DeliveryDistributionSpec
    spoilage_distribution: SpoilageDistributionSpec
    price_distribution: PriceDistributionSpec
    resource_names: tuple[str, ...]
    kernel_id: str | None = None

    def __init__(self, source: Mapping[str, Any]) -> None:
        source = require_json_mapping(source, name="ProbabilityDist input")
        supplier = SupplierSpec(source)
        operating = require_mapping(
            source,
            "operating_conditions",
            path="environment",
        )
        delivery = require_mapping(
            operating,
            "delivery",
            path="environment.operating_conditions",
        )
        spoilage = require_mapping(
            operating,
            "spoilage",
            path="environment.operating_conditions",
        )
        prices = require_mapping(
            operating,
            "prices",
            path="environment.operating_conditions",
        )
        kernel_id = operating.get("kernel_id")
        if kernel_id is not None:
            kernel_id = require_nonempty_string(kernel_id, name="kernel_id")

        object.__setattr__(self, "delivery_distribution", _parse_delivery(delivery))
        object.__setattr__(self, "spoilage_distribution", _parse_spoilage(spoilage))
        object.__setattr__(
            self,
            "price_distribution",
            _parse_prices(prices, expected_resources=supplier.resources),
        )
        object.__setattr__(self, "resource_names", supplier.resources)
        object.__setattr__(self, "kernel_id", kernel_id)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ProbabilityDist":
        """Build from an already parsed environment JSON object."""
        return cls(source)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProbabilityDist":
        """Build from an environment JSON file."""
        return cls(load_json_object(path, name="Environment JSON"))


def build_probability_dist(environment: Mapping[str, Any]) -> ProbabilityDist:
    """Parse declarative probability specifications from environment JSON."""
    return ProbabilityDist(environment)
