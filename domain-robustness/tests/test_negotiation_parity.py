from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from _support import NEGOTIATION_ROOT, load_module
from negotiation.logic.negotiation import run_negotiation as run_outside


domain_protocol = load_module(
    "domain_robustness_negotiation_protocol",
    NEGOTIATION_ROOT / "protocol.py",
)
domain_run = load_module(
    "domain_robustness_negotiation_run",
    NEGOTIATION_ROOT / "run.py",
)


ROLE_NAMES = {
    "hotel-cleaning": ("hotel owner", "cleaning provider"),
    "ai-model-hosting": ("AI startup founder", "AI service provider"),
}


class FakeModel:
    last_thought_trace = None


class ScriptedAgent:
    def __init__(self, name: str, outputs: list[str]) -> None:
        self.name = name
        self.outputs = iter(outputs)
        self.calls: list[str] = []
        self.observations: list[str] = []

    def act(self, action_spec) -> str:
        self.calls.append(action_spec.call_to_action)
        return next(self.outputs)

    def observe(self, observation: str) -> None:
        self.observations.append(observation)


def _normalize(domain: str, text: str) -> str:
    customer, caterer = ROLE_NAMES[domain]
    replacements = (
        (customer.title(), "Customer"),
        (caterer.title(), "Caterer"),
        (customer, "customer"),
        (caterer, "caterer"),
        ("cleaning-service and payment", "delivery and payment"),
        ("service and payment", "delivery and payment"),
    )
    normalized = text
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return normalized


class NegotiationParityTests(unittest.TestCase):
    def _run_pair(
        self,
        domain: str,
        customer_outputs: list[str],
        caterer_outputs: list[str],
        *,
        max_rounds: int,
    ):
        models = [FakeModel(), FakeModel()]
        outside_agents = [
            ScriptedAgent("customer", list(customer_outputs)),
            ScriptedAgent("caterer", list(caterer_outputs)),
        ]
        names = ROLE_NAMES[domain]
        domain_agents = [
            ScriptedAgent(names[0], list(customer_outputs)),
            ScriptedAgent(names[1], list(caterer_outputs)),
        ]
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            outside_result = run_outside(
                outside_agents,
                models,
                max_rounds=max_rounds,
                log_path=root / "outside.log",
                contract_path=root / "outside-contract.txt",
            )
            domain_result = domain_protocol.run_negotiation(
                domain,
                domain_agents,
                models,
                max_rounds=max_rounds,
                log_path=root / "domain.log",
                contract_path=root / "domain-contract.txt",
                debug_json=False,
                record_llm_call=lambda *_args, **_kwargs: None,
            )
        return outside_result, domain_result, outside_agents, domain_agents

    def test_proposal_and_acceptance_match_for_every_domain(self) -> None:
        contract = "Service A: 2 units\nWeek 1 payment: $10"
        customer = [
            '{"response":"What terms work?"}',
            '{"response":"Agreed.","accept":true}',
        ]
        caterer = [
            '{"response":"Try this.","contract":' + json.dumps(contract) + "}"
        ]
        for domain in ROLE_NAMES:
            with self.subTest(domain=domain):
                outside, adapted, outside_agents, domain_agents = self._run_pair(
                    domain,
                    customer,
                    caterer,
                    max_rounds=2,
                )
                self.assertEqual(adapted, outside)
                self.assertEqual(
                    [_normalize(domain, call) for agent in domain_agents for call in agent.calls],
                    [call for agent in outside_agents for call in agent.calls],
                )

    def test_malformed_json_cannot_trigger_termination(self) -> None:
        malformed = '{"response":"leaving"} trailing {"terminate":true}'
        for domain in ROLE_NAMES:
            with self.subTest(domain=domain):
                outside, adapted, _outside_agents, _domain_agents = self._run_pair(
                    domain,
                    [malformed],
                    ['{"response":"Still negotiating."}'],
                    max_rounds=1,
                )
                self.assertEqual(adapted, outside)
                self.assertTrue(adapted[2]["valid"])
                self.assertEqual(adapted[2]["turns_taken"], 2)

    def test_whitespace_contract_cannot_be_accepted(self) -> None:
        customer = [
            '{"response":"Terms?"}',
            '{"response":"I accept.","accept":true}',
        ]
        caterer = [
            '{"response":"Offer.","contract":"   "}',
            '{"response":"No contract yet."}',
        ]
        for domain in ROLE_NAMES:
            with self.subTest(domain=domain):
                outside, adapted, _outside_agents, _domain_agents = self._run_pair(
                    domain,
                    customer,
                    caterer,
                    max_rounds=2,
                )
                self.assertEqual(adapted, outside)
                self.assertEqual(adapted[:2], (None, False))

    def test_every_prompt_uses_the_configured_horizon(self) -> None:
        for domain in ROLE_NAMES:
            for horizon in (1, 7, 50):
                with self.subTest(domain=domain, horizon=horizon):
                    players = domain_run._players(domain, "control", horizon)
                    for player in players:
                        self.assertIn(f"{horizon} rounds", player["goal"])
                        if horizon != 50:
                            self.assertNotIn("50 rounds", player["goal"])

if __name__ == "__main__":
    unittest.main()
