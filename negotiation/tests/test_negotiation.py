from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_utils.llm.providers import EmptyModelResponseError
from negotiation.logic.negotiation import run_negotiation


class _FakeModel:
    last_thought_trace = None


class _FakeAgent:
    def __init__(self, name: str, outputs: list[str]) -> None:
        self.name = name
        self._outputs = iter(outputs)
        self.observations: list[str] = []
        self.action_specs = []

    def act(self, action_spec):
        self.action_specs.append(action_spec)
        return next(self._outputs)

    def observe(self, text: str) -> None:
        self.observations.append(text)


class _BlankAgent(_FakeAgent):
    def __init__(self, name: str, error: Exception) -> None:
        super().__init__(name, [])
        self._error = error

    def act(self, _action_spec):
        raise self._error


class NegotiationProtocolTests(unittest.TestCase):
    def test_second_blank_marks_run_invalid_without_broadcasting(self) -> None:
        error = EmptyModelResponseError(
            "anthropic",
            "claude-test",
            {"reason": "empty_response_after_retry"},
        )
        agents = [_BlankAgent("customer", error), _FakeAgent("caterer", [])]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_negotiation(
                agents,
                [_FakeModel(), _FakeModel()],
                max_rounds=1,
                log_path=Path(temp_dir) / "log.txt",
                contract_path=Path(temp_dir) / "contract.txt",
            )

        _contract, accepted, status = result
        self.assertFalse(accepted)
        self.assertFalse(status["valid"])
        self.assertEqual(status["invalid_reason"], "empty_model_response")
        self.assertEqual(agents[0].observations, [])
        self.assertEqual(agents[1].observations, [])

    def test_proposal_and_acceptance_close_the_exact_contract(self) -> None:
        contract = "Pizza: $10\nRound 1: 1 Pizza\nWeek 1 payment: $10"
        agents = [
            _FakeAgent(
                "customer",
                [
                    '{"response":"What are your terms?"}',
                    '{"response":"I accept.","accept":true}',
                ],
            ),
            _FakeAgent(
                "caterer",
                [
                    '{"response":"Here is my offer.","contract":'
                    + json.dumps(contract)
                    + "}"
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_negotiation(
                agents,
                [_FakeModel(), _FakeModel()],
                max_rounds=2,
                log_path=Path(temp_dir) / "log.txt",
                contract_path=Path(temp_dir) / "contract.txt",
            )

        result_contract, accepted, _status = result
        self.assertTrue(accepted)
        self.assertEqual(result_contract, contract)
        self.assertIn(
            "<<< BEGIN CONTRACT >>>",
            agents[0].action_specs[1].call_to_action,
        )
        self.assertTrue(
            all("[CONTRACT PROPOSAL by caterer]" in agent.observations[1] for agent in agents)
        )

    def test_whitespace_contract_cannot_be_proposed_or_accepted(self) -> None:
        agents = [
            _FakeAgent(
                "customer",
                [
                    '{"response":"What are your terms?"}',
                    '{"response":"I accept.","accept":true}',
                ],
            ),
            _FakeAgent(
                "caterer",
                [
                    '{"response":"Here is my offer.","contract":"   "}',
                    '{"response":"There is no written contract yet."}',
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "log.txt"
            contract_path = Path(temp_dir) / "contract.txt"
            result = run_negotiation(
                agents,
                [_FakeModel(), _FakeModel()],
                max_rounds=2,
                log_path=log_path,
                contract_path=contract_path,
            )
            log = log_path.read_text()

            self.assertFalse(contract_path.exists())

        result_contract, accepted, status = result
        self.assertTrue(status["valid"])
        self.assertFalse(accepted)
        self.assertIsNone(result_contract)
        self.assertIn("ACCEPT DEFERRED", log)
        self.assertNotIn("[CONTRACT by caterer]", log)


if __name__ == "__main__":
    unittest.main()
