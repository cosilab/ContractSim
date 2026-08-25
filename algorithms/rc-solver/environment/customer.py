"""Customer-side environment specification.

The current environment JSON keeps customer data in two sections:
``base_game.customer`` and ``customer_preference``.  ``CustomerSpec`` provides
one validated, immutable view over those legacy sections without changing the
solver's existing configuration format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .json_input import (
    immutable_integer_mapping,
    load_json_object,
    require_finite_number,
    require_json_mapping,
    require_mapping,
)


@dataclass(frozen=True, init=False)
class CustomerSpec:
    """Immutable customer parameters built only from environment JSON."""

    starting_budget: float
    private_values: Mapping[str, int]
    minimum_products_per_production_round: Mapping[str, int]
    substitution_rules: Mapping[str, int]

    def __init__(self, source: Mapping[str, Any]) -> None:
        values = self._parse_mapping(source)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        starting_budget = require_finite_number(
            self.starting_budget,
            name="starting_budget",
            minimum=0,
        )
        private_values = immutable_integer_mapping(
            self.private_values,
            name="private_values",
            minimum=0,
        )
        minimum_products = immutable_integer_mapping(
            self.minimum_products_per_production_round,
            name="minimum_products_per_production_round",
            minimum=0,
        )
        substitution_rules = immutable_integer_mapping(
            self.substitution_rules,
            name="substitution_rules",
            minimum=1,
        )

        product_names = set(private_values)
        for name, mapping in (
            ("minimum_products_per_production_round", minimum_products),
            ("substitution_rules", substitution_rules),
        ):
            if set(mapping) != product_names:
                raise ValueError(
                    f"{name} products must match private_values products: "
                    f"expected {sorted(product_names)}, got {sorted(mapping)}"
                )

        object.__setattr__(self, "starting_budget", starting_budget)
        object.__setattr__(self, "private_values", private_values)
        object.__setattr__(
            self,
            "minimum_products_per_production_round",
            minimum_products,
        )
        object.__setattr__(
            self,
            "substitution_rules",
            substitution_rules,
        )

    @staticmethod
    def _parse_mapping(environment: Mapping[str, Any]) -> dict[str, Any]:
        environment = require_json_mapping(environment, name="CustomerSpec input")
        base_game = require_mapping(environment, "base_game", path="environment")
        customer = require_mapping(
            base_game,
            "customer",
            path="environment.base_game",
        )
        preference = require_mapping(
            environment,
            "customer_preference",
            path="environment",
        )

        try:
            starting_budget = customer["starting_budget"]
            minimum_products = customer[
                "minimum_dishes_per_production_round"
            ]
            private_values = preference["private_values"]
            substitution_rules = preference[
                "substitution_soup_equivalents"
            ]
        except KeyError as exc:
            raise ValueError(f"Missing customer environment field: {exc.args[0]}") from exc

        if not isinstance(minimum_products, Mapping):
            raise ValueError(
                "environment.base_game.customer."
                "minimum_dishes_per_production_round must be a mapping"
            )
        if not isinstance(private_values, Mapping):
            raise ValueError(
                "environment.customer_preference.private_values must be a mapping"
            )
        if not isinstance(substitution_rules, Mapping):
            raise ValueError(
                "environment.customer_preference."
                "substitution_soup_equivalents must be a mapping"
            )

        return {
            "starting_budget": starting_budget,
            "private_values": private_values,
            "minimum_products_per_production_round": minimum_products,
            "substitution_rules": substitution_rules,
        }

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "CustomerSpec":
        """Build from an already parsed environment JSON object."""
        return cls(source)

    @classmethod
    def from_json(cls, path: str | Path) -> "CustomerSpec":
        """Load customer parameters from an environment JSON file."""
        return cls(load_json_object(path, name="Environment JSON"))
