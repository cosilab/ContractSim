"""Preserved domain CTAs over the shared performance engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from concordia.typing import entity as entity_lib

from domains import DOMAINS
from performance.logic.responses import parse_action_response, strip_name_prefix
from domain import format_money


DISHES = ("A", "B", "C")
INGS = ("X", "Y", "Z")
MAX_RETRIES = 3


class Actions:
    def __init__(self, domain: str, record_llm_call) -> None:
        self.domain = domain
        self.spec = DOMAINS[domain]
        self.record_llm_call = record_llm_call
        self.services = self.spec["services"]
        self.materials = self.spec["materials"]
        if domain == "hotel-cleaning":
            self.material_keys = {"X": "buy_Detergent", "Y": "buy_Disinfectant", "Z": "buy_Wet_Wipes"}
            self.service_keys = {d: f"perform_{self.services[d].replace('-', '_').replace(' ', '_')}" for d in DISHES}
        else:
            self.material_keys = {"X": "reserve_CPU", "Y": "reserve_Memory", "Z": "reserve_GPU"}
            self.service_keys = {d: f"provide_{self.services[d].replace('-', '_').replace(' ', '_')}" for d in DISHES}

    def _request(
        self,
        agent: Any,
        cta: str,
        required_keys: Sequence[str],
        fallback: dict[str, Any],
        model: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(MAX_RETRIES):
            call_to_action = (
                cta if attempt == 0 else
                "Your previous response wasn't valid JSON with keys "
                f"{list(required_keys)}. Respond ONLY with the requested JSON. "
                f"Original request: {cta}"
            )
            raw = agent.act(entity_lib.ActionSpec(
                call_to_action=call_to_action,
                output_type=entity_lib.OutputType.FREE,
                tag="performance_action",
            ))
            parser_input = strip_name_prefix(agent.name, raw)
            parsed = parse_action_response(parser_input, required_keys)
            self.record_llm_call(
                agent.name, model, raw_output=raw, parser_input=parser_input,
                parsed_json=parsed,
                parse_meta={"parsed": parsed is not None, "attempt": attempt + 1},
                context={**context, "call_to_action": call_to_action},
            )
            if parsed is not None:
                return parsed
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}: bad JSON from {agent.name} — {parser_input[:120]}]")
        print(f"  [JSON parse failed {MAX_RETRIES} times — using fallback: {fallback}]")
        return dict(fallback)

    def money(self, value: Any) -> str:
        return format_money(self.domain, value)

    def format_services(self, values: dict[str, int], names: bool = False) -> str:
        if names:
            return ", ".join(f"{values[d]} {self.services[d]}" for d in DISHES)
        return ", ".join(f"{values[d]}{d}" for d in DISHES)

    def format_materials(self, values: dict[str, int]) -> str:
        return ", ".join(f"{values[i]} {self.materials[i]}" for i in INGS)

    def format_prices(self, game: Any) -> str:
        return ", ".join(
            f"{self.materials[i]} {self.money(game.ingredient_prices[i])}/unit" for i in INGS
        )

    def _shortages(self, game: Any, production: dict[str, int]) -> list[str]:
        needed = {i: 0 for i in INGS}
        for dish in DISHES:
            for ingredient, quantity in game.environment.recipes[dish].items():
                needed[ingredient] += production[dish] * quantity
        return [
            f"{self.materials[i]}: need {needed[i]} but only have {game.inventory[i]}"
            for i in INGS if needed[i] > game.inventory[i]
        ]

    def decide_payment(self, game: Any, week: int, agent: Any, total_paid: float, model: Any) -> int:
        counterparty = "Cleaning Provider" if self.domain == "hotel-cleaning" else "AI Service Provider"
        multiple = self.spec["money_scale"]
        cta = (
            f"── Week {week} of {game.total_weeks} (PAYMENT WEEK) ──\n"
            f"Total paid so far: {self.money(total_paid)}\n"
            f"Remaining budget: {self.money(game.customer_budget)}\n"
            f"In light of the above, make a payment to the {counterparty}. "
            f"The payment must be a non-negative multiple of ${multiple:,}, including $0, "
            "and no greater than your remaining budget. "
            'Respond ONLY with JSON: {"payment": <int>}'
        )
        rejection = ""
        for outer in range(MAX_RETRIES):
            context = {
                "phase": "performance", "action": "payment",
                "round": week, "week": week,
            }
            context.update(domain=self.domain, monetary_scale=self.spec["money_scale"])
            response = self._request(
                agent, cta if outer == 0 else rejection + "\n" + cta,
                ["payment"], {"payment": 0}, model, context,
            )
            raw = response["payment"]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                displayed = -1.0
            else:
                displayed = float(raw)
            scale = self.spec["money_scale"]
            if math.isfinite(displayed) and displayed.is_integer():
                displayed_int = int(displayed)
                canonical = displayed_int // scale
                if (
                    displayed_int >= 0
                    and displayed_int % scale == 0
                    and canonical <= game.customer_budget
                ):
                    return canonical
            rejection = (
                "Your payment was REJECTED — payments must be non-negative "
                f"multiples of ${self.spec['money_scale']:,}, including $0, and no greater than your "
                f"remaining budget of {self.money(game.customer_budget)}. Try again."
            )
        actor = "Hotel Owner" if self.domain == "hotel-cleaning" else "AI Startup Founder"
        print(f"  [payment defaulting to $0 — {actor} did not provide a valid multiple of ${self.spec['money_scale']:,}]")
        return 0

    def decide_purchases(self, game: Any, week: int, agent: Any, model: Any) -> dict[str, int]:
        keys = list(self.material_keys.values())
        example = ", ".join(f'"{key}": <int>' for key in keys)
        if self.domain == "hotel-cleaning":
            cta = (
                f"── Week {week} of {game.total_weeks} (CLEANING WEEK — buying) ──\n"
                f"Balance: {self.money(game.supplier_balance)}\n"
                f"Inventory: {self.format_materials(game.inventory)}\n"
                f"Market prices: {self.format_prices(game)}\n"
                "Buy cleaning supplies to perform cleaning services.\n"
                f"Respond ONLY with JSON: {{{example}}}"
            )
        else:
            cta = (
                f"── Week {week} of {game.total_weeks} (SERVICE WEEK — reserving) ──\n"
                f"Balance: {self.money(game.supplier_balance)}\n"
                f"Available resources: {self.format_materials(game.inventory)}\n"
                f"Market prices: {self.format_prices(game)}\n"
                "Reserve resources to provide AI services.\n"
                f"Respond ONLY with JSON: {{{example}}}"
            )
        cost = 0.0
        for outer in range(MAX_RETRIES):
            if outer == 0:
                request = cta
            elif self.domain == "hotel-cleaning":
                request = (
                    f"Your order was REJECTED — it costs {self.money(cost)} but you only have "
                    f"{self.money(game.supplier_balance)}. Order fewer cleaning supplies.\n{cta}"
                )
            else:
                request = (
                    f"Your reservation was REJECTED — it costs {self.money(cost)} but you only have "
                    f"{self.money(game.supplier_balance)}. Reserve fewer resources.\n{cta}"
                )
            context = {
                "phase": "performance", "action": "purchase",
                "round": week, "week": week,
            }
            context.update(domain=self.domain, monetary_scale=self.spec["money_scale"])
            response = self._request(agent, request, keys, {key: 0 for key in keys}, model, context)
            try:
                purchase = {
                    i: min(max(0, int(response[self.material_keys[i]])), game.environment.max_order)
                    for i in INGS
                }
            except (TypeError, ValueError, OverflowError):
                purchase = {i: 0 for i in INGS}
            cost = game.purchase_cost(purchase)
            if cost <= game.supplier_balance:
                return purchase
            noun = "order" if self.domain != "ai-model-hosting" else "reservation"
            print(f"  [retry {outer + 1}/{MAX_RETRIES}: {noun} costs {self.money(cost)} but balance is {self.money(game.supplier_balance)}]")
        message = {
            "hotel-cleaning": "purchases defaulting to 0 — order was not affordable",
            "ai-model-hosting": "reservations defaulting to 0 — reservation was not affordable",
        }[self.domain]
        print(f"  [{message}]")
        return {i: 0 for i in INGS}

    def decide_production(
        self, game: Any, week: int, agent: Any, model: Any, *, received: dict[str, int]
    ) -> dict[str, int]:
        keys = list(self.service_keys.values())
        example = ", ".join(f'"{key}": <int>' for key in keys)
        if self.domain == "hotel-cleaning":
            cta = (
                f"── Cleaning services (Week {week} of {game.total_weeks}) ──\n"
                f"Balance: {self.money(game.supplier_balance)}\n"
                f"Cleaning supplies received: {self.format_materials(received)}\n"
                f"Updated inventory (after cap): {self.format_materials(game.inventory)}\n"
                "Perform cleaning services using your current cleaning supplies.\n"
                f"Respond ONLY with JSON: {{{example}}}"
            )
        else:
            cta = (
                f"── Service provision (Week {week} of {game.total_weeks}) ──\n"
                f"Balance: {self.money(game.supplier_balance)}\n"
                f"Reserved resources received: {self.format_materials(received)}\n"
                f"Updated available resources (after cap): {self.format_materials(game.inventory)}\n"
                "Provide AI services using your current resources.\n"
                f"Respond ONLY with JSON: {{{example}}}"
            )
        shortages: list[str] = []
        for outer in range(MAX_RETRIES):
            if outer == 0:
                request = cta
            elif self.domain == "hotel-cleaning":
                request = "Your cleaning-service decision was REJECTED — not enough cleaning supplies.\n" + "\n".join(shortages) + f"\nTry again.\n{cta}"
            else:
                request = "Your service-provision decision was REJECTED — not enough resources.\n" + "\n".join(shortages) + f"\nTry again.\n{cta}"
            context = {
                "phase": "performance", "action": "production",
                "round": week, "week": week,
            }
            context.update(domain=self.domain, monetary_scale=self.spec["money_scale"])
            response = self._request(agent, request, keys, {key: 0 for key in keys}, model, context)
            try:
                production = {d: max(0, int(response[self.service_keys[d]])) for d in DISHES}
            except (TypeError, ValueError, OverflowError):
                production = {d: 0 for d in DISHES}
            shortages = self._shortages(game, production)
            if not shortages:
                return production
            message = {
                "hotel-cleaning": "cleaning services exceed supply inventory",
                "ai-model-hosting": "AI services exceed available resources",
            }[self.domain]
            print(f"  [retry {outer + 1}/{MAX_RETRIES}: {message}]")
        message = {
            "hotel-cleaning": "cleaning services defaulting to 0 — insufficient supplies",
            "ai-model-hosting": "service provision defaulting to 0 — insufficient resources",
        }[self.domain]
        print(f"  [{message}]")
        return {d: 0 for d in DISHES}
