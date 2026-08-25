from __future__ import annotations

import unittest

from performance.engine.environment import EnvironmentEngine
from performance.logic.performance import (
    _request_json_action,
    customer_decide_payment,
    supplier_decide_production,
    supplier_decide_purchases,
)
from performance.logic.responses import parse_action_response, strip_name_prefix


class FakeAgent:
    def __init__(self, name: str, responses: list[str]) -> None:
        self.name = name
        self.responses = iter(responses)
        self.calls: list[str] = []

    def act(self, action_spec) -> str:
        self.calls.append(action_spec.call_to_action)
        return next(self.responses)


class PerformanceResponseTests(unittest.TestCase):
    def test_parser_accepts_plain_fenced_and_embedded_json(self) -> None:
        required = ("payment",)
        self.assertEqual(
            parse_action_response('{"payment": 10}', required),
            {"payment": 10},
        )
        self.assertEqual(
            parse_action_response('```json\n{"payment": 9}\n```', required),
            {"payment": 9},
        )
        self.assertEqual(
            parse_action_response('Decision: {"payment": 8}', required),
            {"payment": 8},
        )
        self.assertIsNone(parse_action_response('{"amount": 8}', required))

    def test_json_request_retries_and_strips_the_agent_name(self) -> None:
        agent = FakeAgent(
            "customer",
            ["not json", 'customer: {"payment": 7}'],
        )
        result = _request_json_action(
            agent,
            "Pay now",
            ("payment",),
            fallback={"payment": 0},
        )

        self.assertEqual(result, {"payment": 7})
        self.assertEqual(len(agent.calls), 2)
        self.assertIn("previous response wasn't valid JSON", agent.calls[1])
        self.assertEqual(strip_name_prefix("customer", "customer: hello"), "hello")


class PerformanceActionTests(unittest.TestCase):
    def test_customer_payment_retries_out_of_budget_amount(self) -> None:
        game = EnvironmentEngine(1)
        agent = FakeAgent(
            "customer",
            ['{"payment": 999}', '{"payment": 10}'],
        )

        self.assertEqual(customer_decide_payment(game, 1, agent, 0), 10)
        self.assertIn("payment was REJECTED", agent.calls[1])

    def test_supplier_purchase_and_production_actions_use_live_state(self) -> None:
        game = EnvironmentEngine(1)
        game.apply_payment(10)
        supplier = FakeAgent(
            "supplier",
            [
                '{"buy_Mozzarella": 2, "buy_Basil": 2, "buy_Tomato": 2}',
                (
                    '{"produce_Margherita_Pizza": 1, '
                    '"produce_Pesto_Pasta": 1, "produce_Tomato_Soup": 1}'
                ),
            ],
        )

        purchase = supplier_decide_purchases(game, 2, supplier)
        self.assertEqual(purchase, {"X": 2, "Y": 2, "Z": 2})
        game.apply_purchase(purchase)
        production = supplier_decide_production(game, 2, supplier)
        self.assertEqual(production, {"A": 1, "B": 1, "C": 1})
        self.assertIn("Market prices", supplier.calls[0])
        self.assertIn("Inventory", supplier.calls[1])


if __name__ == "__main__":
    unittest.main()
