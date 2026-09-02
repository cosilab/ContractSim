from __future__ import annotations

import json
import unittest

from _support import PERFORMANCE_ROOT, load_module
from performance.baselines.customer import CustomerRC, CustomerRCC, CustomerRE
from performance.baselines.supplier import SupplierRC, SupplierRCC, SupplierRE
from performance.engine.environment import EnvironmentEngine
from performance.engine import run_performance
from performance.logic import (
    customer_decide_payment,
    supplier_decide_production,
    supplier_decide_purchases,
)
from performance.orchestration.worker import _LLMCustomer, _LLMSupplier
from performance.tests.helpers import sample_terms


domain_actions = load_module(
    "domain_robustness_performance_actions",
    PERFORMANCE_ROOT / "actions.py",
)
domain_sides = load_module(
    "domain_robustness_performance_sides",
    PERFORMANCE_ROOT / "sides.py",
)
domain_contracts = load_module(
    "domain_robustness_performance_domain",
    PERFORMANCE_ROOT / "domain.py",
)
def json_object(keys: tuple[str, str, str], values: tuple[int, int, int]) -> str:
    return json.dumps(dict(zip(keys, values, strict=True)), separators=(",", ":"))


class ScriptedAgent:
    def __init__(self, name: str, responses: list[str]) -> None:
        self.name = name
        self.responses = iter(responses)
        self.calls: list[str] = []
        self.observations: list[str] = []

    def act(self, action_spec) -> str:
        self.calls.append(action_spec.call_to_action)
        return next(self.responses)

    def observe(self, observation: str) -> None:
        self.observations.append(observation)


class PerformanceParityTests(unittest.TestCase):
    def test_translated_contracts_use_domain_service_wording(self) -> None:
        source = (
            "Parties: Customer and Caterer.\n"
            "Delivery obligations:\n"
            "Week 2: 1 Margherita Pizza.\n"
            "After delivering the minimum, the delivery obligation is determined "
            "from actual physical deliveries."
        )
        expected_action = {
            "hotel-cleaning": "After performing the minimum",
            "ai-model-hosting": "After providing the minimum",
        }
        for domain, action_wording in expected_action.items():
            with self.subTest(domain=domain):
                translated = domain_contracts._adapt_contract(domain, source)
                self.assertIn("Service obligations:", translated)
                self.assertIn("service obligation", translated)
                self.assertIn("actual provided services", translated)
                self.assertIn(action_wording, translated)
                self.assertNotIn("Delivery obligations", translated)
                self.assertNotIn("delivery obligation", translated)
                self.assertNotIn("actual physical deliveries", translated)
                self.assertNotIn("After delivering", translated)

    def test_translated_payments_normalize_to_canonical_units(self) -> None:
        cases = (("hotel-cleaning", 100, 1_000, 10), ("ai-model-hosting", 1_000, 10_000, 10))
        for domain, _scale, displayed, canonical in cases:
            with self.subTest(domain=domain):
                actions = domain_actions.Actions(domain, lambda *_args, **_kwargs: None)
                game = EnvironmentEngine(5)
                agent = ScriptedAgent("payer", [f'{{"payment":{displayed}}}'])
                self.assertEqual(
                    actions.decide_payment(game, 1, agent, 0, None),
                    canonical,
                )

    def test_translated_domains_match_payment_retries_after_money_normalization(self) -> None:
        for domain, scale in (("hotel-cleaning", 100), ("ai-model-hosting", 1_000)):
            with self.subTest(domain=domain):
                outside_agent = ScriptedAgent(
                    "customer",
                    ["bad json", '{"payment":1.5}', '{"payment":10}'],
                )
                domain_agent = ScriptedAgent(
                    "payer",
                    [
                        "bad json",
                        f'{{"payment":{1.5 * scale}}}',
                        f'{{"payment":{10 * scale}}}',
                    ],
                )
                outside = customer_decide_payment(
                    EnvironmentEngine(5),
                    1,
                    outside_agent,
                    0,
                )
                adapted = domain_actions.Actions(
                    domain,
                    lambda *_args, **_kwargs: None,
                ).decide_payment(
                    EnvironmentEngine(5),
                    1,
                    domain_agent,
                    0,
                    None,
                )
                self.assertEqual(adapted, outside)
                self.assertEqual(len(domain_agent.calls), len(outside_agent.calls))

    def test_translated_domains_match_affordability_and_feasibility_retries(self) -> None:
        purchase_keys = {
            "hotel-cleaning": ("buy_Detergent", "buy_Disinfectant", "buy_Wet_Wipes"),
            "ai-model-hosting": ("reserve_CPU", "reserve_Memory", "reserve_GPU"),
        }
        production_keys = {
            "hotel-cleaning": (
                "perform_Conference_Room_Cleaning",
                "perform_King_Sized_Room_Cleaning",
                "perform_Single_Room_Cleaning",
            ),
            "ai-model-hosting": (
                "provide_Fine_tuning",
                "provide_Model_hosting",
                "provide_Inference",
            ),
        }
        outside_purchase_responses = [
            '{"buy_Mozzarella":12,"buy_Basil":12,"buy_Tomato":12}',
            '{"buy_Mozzarella":0,"buy_Basil":0,"buy_Tomato":0}',
        ]
        outside_production_responses = [
            '{"produce_Margherita_Pizza":1,"produce_Pesto_Pasta":1,"produce_Tomato_Soup":1}',
            '{"produce_Margherita_Pizza":0,"produce_Pesto_Pasta":0,"produce_Tomato_Soup":0}',
        ]
        for domain in purchase_keys:
            with self.subTest(domain=domain):
                actions = domain_actions.Actions(domain, lambda *_args, **_kwargs: None)
                domain_purchase_responses = [
                    json_object(purchase_keys[domain], (12, 12, 12)),
                    json_object(purchase_keys[domain], (0, 0, 0)),
                ]
                outside_agent = ScriptedAgent("supplier", outside_purchase_responses)
                domain_agent = ScriptedAgent("provider", domain_purchase_responses)
                outside_game = EnvironmentEngine(5)
                domain_game = EnvironmentEngine(5)
                outside_purchase = supplier_decide_purchases(outside_game, 2, outside_agent)
                domain_purchase = actions.decide_purchases(domain_game, 2, domain_agent, None)
                self.assertEqual(domain_purchase, outside_purchase)
                self.assertEqual(len(domain_agent.calls), len(outside_agent.calls))

                outside_agent = ScriptedAgent("supplier", outside_production_responses)
                domain_agent = ScriptedAgent(
                    "provider",
                    [
                        json_object(production_keys[domain], (1, 1, 1)),
                        json_object(production_keys[domain], (0, 0, 0)),
                    ],
                )
                outside_production = supplier_decide_production(
                    outside_game,
                    2,
                    outside_agent,
                    received={"X": 0, "Y": 0, "Z": 0},
                )
                domain_production = actions.decide_production(
                    domain_game,
                    2,
                    domain_agent,
                    None,
                    received={"X": 0, "Y": 0, "Z": 0},
                )
                self.assertEqual(domain_production, outside_production)
                self.assertEqual(len(domain_agent.calls), len(outside_agent.calls))

    def test_domain_baselines_are_the_shared_classes(self) -> None:
        self.assertEqual(
            domain_sides.CUSTOMER_BASELINES,
            {"rc": CustomerRC, "re": CustomerRE, "rcc": CustomerRCC},
        )
        self.assertEqual(
            domain_sides.SUPPLIER_BASELINES,
            {"rc": SupplierRC, "re": SupplierRE, "rcc": SupplierRCC},
        )

    def test_complete_translated_llm_trajectories_match_canonical_results(self) -> None:
        specifications = {
            "hotel-cleaning": {
                "scale": 100,
                "purchase": '{"buy_Detergent":0,"buy_Disinfectant":0,"buy_Wet_Wipes":0}',
                "production": (
                    '{"perform_Conference_Room_Cleaning":0,'
                    '"perform_King_Sized_Room_Cleaning":0,'
                    '"perform_Single_Room_Cleaning":0}'
                ),
            },
            "ai-model-hosting": {
                "scale": 1_000,
                "purchase": '{"reserve_CPU":0,"reserve_Memory":0,"reserve_GPU":0}',
                "production": (
                    '{"provide_Fine_tuning":0,"provide_Model_hosting":0,'
                    '"provide_Inference":0}'
                ),
            },
        }
        outside_supplier_responses = [
            '{"buy_Mozzarella":0,"buy_Basil":0,"buy_Tomato":0}',
            '{"produce_Margherita_Pizza":0,"produce_Pesto_Pasta":0,"produce_Tomato_Soup":0}',
        ] * 5
        for domain, specification in specifications.items():
            for seed in (42, 44):
                with self.subTest(domain=domain, seed=seed):
                    outside_customer = _LLMCustomer(
                        ScriptedAgent("customer", ['{"payment":10}'] * 6),
                        None,
                        lambda *_args, **_kwargs: None,
                        total_weeks=11,
                    )
                    outside_supplier = _LLMSupplier(
                        ScriptedAgent("supplier", outside_supplier_responses),
                        None,
                        lambda *_args, **_kwargs: None,
                    )
                    domain_customer = self._uninitialized_domain_side(
                        domain,
                        "customer",
                        ScriptedAgent(
                            "payer",
                            [f'{{"payment":{10 * specification["scale"]}}}'] * 6,
                        ),
                    )
                    domain_supplier = self._uninitialized_domain_side(
                        domain,
                        "supplier",
                        ScriptedAgent(
                            "provider",
                            [specification["purchase"], specification["production"]] * 5,
                        ),
                    )
                    outside_result = run_performance(
                        customer=outside_customer,
                        supplier=outside_supplier,
                        terms=sample_terms(),
                        environment_id=5,
                        seed=seed,
                    )
                    domain_result = run_performance(
                        customer=domain_customer,
                        supplier=domain_supplier,
                        terms=sample_terms(),
                        environment_id=5,
                        seed=seed,
                    )
                    self.assertEqual(domain_result, outside_result)
                    self.assertEqual(len(domain_customer._agent.calls), 6)
                    self.assertEqual(len(domain_supplier._agent.calls), 10)
                    self.assertEqual(len(domain_customer._agent.observations), 11)
                    self.assertEqual(len(domain_supplier._agent.observations), 6)

    def _uninitialized_domain_side(
        self,
        domain: str,
        role: str,
        agent: ScriptedAgent,
    ):
        side = object.__new__(domain_sides.LLMSide)
        side.domain = domain
        side.canonical_role = role
        side.total_paid = 0.0
        side.delivered = {"A": 0, "B": 0, "C": 0}
        side._agent = agent
        side._model = None
        side.actions = domain_actions.Actions(
            domain,
            lambda *_args, **_kwargs: None,
        )
        return side


if __name__ == "__main__":
    unittest.main()
