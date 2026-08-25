"""Supplier-side environment specification.

The existing environment JSON splits supplier data between ``base_game`` and
``caterer_capital``.  ``SupplierSpec`` combines those sections into one
validated, immutable domain object without changing the solver's current input
format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .json_input import (
    load_json_object,
    require_integer,
    require_json_mapping,
    require_mapping,
)


def _recipes(value: Mapping[str, Any]) -> Mapping[str, Mapping[str, int]]:
    normalized: dict[str, Mapping[str, int]] = {}
    for product, raw_recipe in value.items():
        if not isinstance(product, str) or not product:
            raise ValueError("recipes keys must be non-empty product names")
        if not isinstance(raw_recipe, Mapping):
            raise ValueError(f"recipes[{product!r}] must be a mapping")

        recipe: dict[str, int] = {}
        for resource, raw_quantity in raw_recipe.items():
            if not isinstance(resource, str) or not resource:
                raise ValueError(
                    f"recipes[{product!r}] keys must be non-empty resource names"
                )
            quantity = require_integer(
                raw_quantity,
                name=f"recipes[{product!r}][{resource!r}]",
                minimum=0,
            )
            if quantity:
                recipe[resource] = quantity

        if not recipe:
            raise ValueError(f"recipes[{product!r}] must use at least one resource")
        normalized[product] = MappingProxyType(recipe)

    if not normalized:
        raise ValueError("recipes must contain at least one product")
    return MappingProxyType(normalized)


@dataclass(frozen=True, init=False)
class SupplierSpec:
    """Immutable supplier parameters built only from environment JSON."""

    starting_capital: int
    number_of_capital_states: int
    recipes: Mapping[str, Mapping[str, int]]
    max_inventory_per_resource: int
    max_order_per_resource_per_round: int

    def __init__(self, source: Mapping[str, Any]) -> None:
        values = self._parse_mapping(source)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        starting_capital = require_integer(
            self.starting_capital,
            name="starting_capital",
            minimum=0,
        )
        number_of_capital_states = require_integer(
            self.number_of_capital_states,
            name="number_of_capital_states",
            minimum=1,
        )
        if starting_capital >= number_of_capital_states:
            raise ValueError(
                "starting_capital must be smaller than number_of_capital_states"
            )

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
        recipes = _recipes(self.recipes)

        object.__setattr__(self, "starting_capital", starting_capital)
        object.__setattr__(
            self,
            "number_of_capital_states",
            number_of_capital_states,
        )
        object.__setattr__(self, "recipes", recipes)
        object.__setattr__(
            self,
            "max_inventory_per_resource",
            max_inventory,
        )
        object.__setattr__(
            self,
            "max_order_per_resource_per_round",
            max_order,
        )

    @property
    def products(self) -> tuple[str, ...]:
        return tuple(self.recipes)

    @property
    def resources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                resource
                for recipe in self.recipes.values()
                for resource in recipe
            )
        )

    @staticmethod
    def _parse_mapping(environment: Mapping[str, Any]) -> dict[str, Any]:
        environment = require_json_mapping(environment, name="SupplierSpec input")
        base_game = require_mapping(environment, "base_game", path="environment")
        inventory = require_mapping(
            base_game,
            "inventory",
            path="environment.base_game",
        )
        capital = require_mapping(
            environment,
            "caterer_capital",
            path="environment",
        )
        recipes = require_mapping(
            base_game,
            "recipes",
            path="environment.base_game",
        )

        try:
            starting_capital = capital["starting_balance"]
            number_of_capital_states = capital["n_budget_states"]
            max_inventory = inventory["max_units_per_ingredient"]
            max_order = inventory["max_order_per_ingredient_per_round"]
        except KeyError as exc:
            raise ValueError(f"Missing supplier environment field: {exc.args[0]}") from exc

        return {
            "starting_capital": starting_capital,
            "number_of_capital_states": number_of_capital_states,
            "recipes": recipes,
            "max_inventory_per_resource": max_inventory,
            "max_order_per_resource_per_round": max_order,
        }

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "SupplierSpec":
        """Build from an already parsed environment JSON object."""
        return cls(source)

    @classmethod
    def from_json(cls, path: str | Path) -> "SupplierSpec":
        """Load supplier parameters from an environment JSON file."""
        return cls(load_json_object(path, name="Environment JSON"))
