"""LLM call-to-actions for contract performance decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from concordia.typing import entity as entity_lib

from .responses import parse_action_response, strip_name_prefix


DISHES = ("A", "B", "C")
INGREDIENTS = ("X", "Y", "Z")
DISH_NAMES = {
    "A": "Margherita Pizza",
    "B": "Pesto Pasta",
    "C": "Tomato Soup",
}
INGREDIENT_NAMES = {"X": "Mozzarella", "Y": "Basil", "Z": "Tomato"}
INGREDIENT_KEYS = {
    ingredient: f"buy_{name}"
    for ingredient, name in INGREDIENT_NAMES.items()
}
DISH_KEYS = {
    dish: f"produce_{name.replace(' ', '_')}"
    for dish, name in DISH_NAMES.items()
}
MAX_RETRIES = 3


def _request_json_action(
    agent: Any,
    call_to_action: str,
    required_keys: Sequence[str],
    *,
    fallback: dict[str, Any],
    model: Any = None,
    record_llm_call: Callable[..., None] | None = None,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request a JSON action using ContractBench's three-attempt policy."""
    for attempt in range(1, MAX_RETRIES + 1):
        cta = (
            call_to_action
            if attempt == 1
            else (
                "Your previous response wasn't valid JSON with keys "
                f"{list(required_keys)}. Respond ONLY with the requested JSON. "
                f"Original request: {call_to_action}"
            )
        )
        raw = agent.act(
            entity_lib.ActionSpec(
                call_to_action=cta,
                output_type=entity_lib.OutputType.FREE,
                tag="performance_action",
            )
        )
        parser_input = strip_name_prefix(agent.name, raw)
        parsed = parse_action_response(parser_input, required_keys)
        if record_llm_call is not None:
            record_llm_call(
                agent.name,
                model,
                raw_output=raw,
                parser_input=parser_input,
                parsed_json=parsed,
                parse_meta={"parsed": parsed is not None, "attempt": attempt},
                context={
                    **(trace_context or {}),
                    "call_to_action": cta,
                },
            )
        if parsed is not None:
            return parsed
    return dict(fallback)


def _format_ingredients(values: dict[str, int]) -> str:
    return ", ".join(
        f"{values[ingredient]} {INGREDIENT_NAMES[ingredient]}"
        for ingredient in INGREDIENTS
    )


def _format_prices(game: Any) -> str:
    return ", ".join(
        f"{INGREDIENT_NAMES[ingredient]} "
        f"${game.ingredient_prices[ingredient]:.2f}/unit"
        for ingredient in INGREDIENTS
    )


def _production_shortages(
    game: Any,
    production: dict[str, int],
) -> list[str]:
    required = {ingredient: 0 for ingredient in INGREDIENTS}
    for dish in DISHES:
        for ingredient, quantity in game.environment.recipes[dish].items():
            required[ingredient] += production[dish] * quantity
    return [
        f"{INGREDIENT_NAMES[ingredient]}: need {required[ingredient]} "
        f"but only have {game.inventory[ingredient]}"
        for ingredient in INGREDIENTS
        if required[ingredient] > game.inventory[ingredient]
    ]


def customer_decide_payment(
    game: Any,
    week: int,
    agent: Any,
    total_paid: float,
    *,
    model: Any = None,
    record_llm_call: Callable[..., None] | None = None,
) -> int:
    """Ask the customer for a valid whole-dollar payment."""
    cta = (
        f"── Week {week} of {game.total_weeks} (PAYMENT WEEK) ──\n"
        f"Total paid so far: ${total_paid:.2f}\n"
        f"Remaining budget: ${game.customer_budget:.2f}\n"
        "In light of the above, make a payment to the Supplier. "
        'Respond ONLY with JSON: {"payment": <int>}'
    )
    rejection = ""
    for attempt in range(1, MAX_RETRIES + 1):
        response = _request_json_action(
            agent,
            cta if attempt == 1 else f"{rejection}\n{cta}",
            ("payment",),
            fallback={"payment": 0},
            model=model,
            record_llm_call=record_llm_call,
            trace_context={
                "phase": "performance",
                "action": "payment",
                "round": week,
                "week": week,
            },
        )
        raw_payment = response["payment"]
        if isinstance(raw_payment, bool) or not isinstance(
            raw_payment,
            (int, float),
        ):
            payment = -1.0
        else:
            payment = float(raw_payment)
        if payment.is_integer() and 0 <= payment <= game.customer_budget:
            return int(payment)
        rejection = (
            "Your payment was REJECTED — payments must be non-negative "
            "whole-dollar amounts no greater than your remaining budget of "
            f"${game.customer_budget:.2f}. Try again."
        )
    return 0


def supplier_decide_purchases(
    game: Any,
    week: int,
    agent: Any,
    *,
    model: Any = None,
    record_llm_call: Callable[..., None] | None = None,
) -> dict[str, int]:
    """Ask the supplier for an affordable ingredient order."""
    keys = tuple(INGREDIENT_KEYS.values())
    example = ", ".join(f'"{key}": <int>' for key in keys)
    cta = (
        f"── Week {week} of {game.total_weeks} (PRODUCTION WEEK — buying) ──\n"
        f"Balance: ${game.supplier_balance:.2f}\n"
        f"Inventory: {_format_ingredients(game.inventory)}\n"
        f"Market prices: {_format_prices(game)}\n"
        "Buy ingredients to produce dishes.\n"
        f"Respond ONLY with JSON: {{{example}}}"
    )
    cost = 0.0
    for attempt in range(1, MAX_RETRIES + 1):
        request = (
            cta
            if attempt == 1
            else (
                f"Your order was REJECTED — it costs ${cost:.2f} but you only "
                f"have ${game.supplier_balance:.2f}. Order fewer ingredients.\n"
                f"{cta}"
            )
        )
        response = _request_json_action(
            agent,
            request,
            keys,
            fallback={key: 0 for key in keys},
            model=model,
            record_llm_call=record_llm_call,
            trace_context={
                "phase": "performance",
                "action": "purchase",
                "round": week,
                "week": week,
            },
        )
        try:
            purchase = {
                ingredient: min(
                    max(0, int(response[INGREDIENT_KEYS[ingredient]])),
                    game.environment.max_order,
                )
                for ingredient in INGREDIENTS
            }
        except (TypeError, ValueError, OverflowError):
            purchase = {ingredient: 0 for ingredient in INGREDIENTS}
        cost = game.purchase_cost(purchase)
        if cost <= game.supplier_balance:
            return purchase
    return {ingredient: 0 for ingredient in INGREDIENTS}


def supplier_decide_production(
    game: Any,
    week: int,
    agent: Any,
    *,
    received: dict[str, int],
    model: Any = None,
    record_llm_call: Callable[..., None] | None = None,
) -> dict[str, int]:
    """Ask the supplier for a production plan feasible from inventory."""
    keys = tuple(DISH_KEYS.values())
    example = ", ".join(f'"{key}": <int>' for key in keys)
    cta = (
        f"── Production (Week {week} of {game.total_weeks}) ──\n"
        f"Balance: ${game.supplier_balance:.2f}\n"
        f"Order received: {_format_ingredients(received)}\n"
        f"Updated inventory (after cap): {_format_ingredients(game.inventory)}\n"
        "Produce dishes using your current inventory.\n"
        f"Respond ONLY with JSON: {{{example}}}"
    )
    shortages: list[str] = []
    for attempt in range(1, MAX_RETRIES + 1):
        request = (
            cta
            if attempt == 1
            else (
                "Your production was REJECTED — not enough ingredients.\n"
                + "\n".join(shortages)
                + f"\nTry again.\n{cta}"
            )
        )
        response = _request_json_action(
            agent,
            request,
            keys,
            fallback={key: 0 for key in keys},
            model=model,
            record_llm_call=record_llm_call,
            trace_context={
                "phase": "performance",
                "action": "production",
                "round": week,
                "week": week,
            },
        )
        try:
            production = {
                dish: max(0, int(response[DISH_KEYS[dish]]))
                for dish in DISHES
            }
        except (TypeError, ValueError, OverflowError):
            production = {dish: 0 for dish in DISHES}
        shortages = _production_shortages(game, production)
        if not shortages:
            return production
    return {dish: 0 for dish in DISHES}
