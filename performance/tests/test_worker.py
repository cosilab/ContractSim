from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llm_utils.llm import EmptyModelResponseError, ProviderConfig
from performance.orchestration.worker import run_worker
from performance.tests.helpers import write_contract


class FakeGeminiModel:
    provider = "gemini"
    model_name = "gemini-test"
    last_thought_trace = "synthetic reasoning"
    last_prompt_cache_usage = {"cached_tokens": 0}
    last_response_diagnostics = {"finish_reasons": ["STOP"]}
    chat_metadata = {"message_count": 0}


class FakeSupplierAgent:
    name = "supplier"

    def act(self, action_spec) -> str:
        if "buy_Mozzarella" in action_spec.call_to_action:
            return '{"buy_Mozzarella": 0, "buy_Basil": 0, "buy_Tomato": 0}'
        return (
            '{"produce_Margherita_Pizza": 0, "produce_Pesto_Pasta": 0, '
            '"produce_Tomato_Soup": 0}'
        )

    def observe(self, _observation: str) -> None:
        return None

    def get_state(self) -> dict:
        return {}


class EmptySupplierAgent(FakeSupplierAgent):
    def act(self, _action_spec) -> str:
        raise EmptyModelResponseError(
            "gemini",
            "gemini-test",
            {"reason": "empty_response_after_retry"},
        )


class BaselineWorkerIntegrationTests(unittest.TestCase):
    def test_rc_customer_and_supplier_complete_a_worker_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_dir = write_contract(root / "contract-1", environment_id=1)
            output_dir = root / "output"
            args = Namespace(
                env=1,
                contract_dir=contract_dir,
                customer="rc",
                supplier="rc",
                seed=42,
                output_dir=output_dir,
                goal_treatment="control",
            )

            self.assertEqual(run_worker(args), 0)
            summary = json.loads(
                (output_dir / "data" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertTrue(summary["valid"])
            self.assertEqual(summary["environment_id"], 1)
            self.assertEqual(summary["total_customer_spent"], 60)
            self.assertEqual(summary["customer_remaining_budget"], 140)
            self.assertEqual(summary["supplier_final_balance"], 40)
            self.assertEqual(summary["delivered"], {"A": 5, "B": 5, "C": 5})
            self.assertEqual(summary["customer_summary"]["baseline"], "rc")
            self.assertEqual(summary["supplier_summary"]["baseline"], "rc")
            self.assertTrue(
                (output_dir / "config" / "performance_players.json").is_file()
            )

    def test_rcc_customer_and_llm_supplier_write_reasoning_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_dir = write_contract(root / "contract-1", environment_id=1)
            output_dir = root / "output"
            agent = FakeSupplierAgent()
            model = FakeGeminiModel()
            runtime = SimpleNamespace(
                agents=(agent,),
                models=(model,),
                configs=(ProviderConfig("gemini", "gemini-test", "high"),),
            )
            args = Namespace(
                env=1,
                contract_dir=contract_dir,
                customer="rcc",
                supplier="gemini",
                seed=42,
                output_dir=output_dir,
                goal_treatment="control",
            )

            with mock.patch(
                "performance.orchestration.worker.build_llm_runtime",
                return_value=runtime,
            ):
                self.assertEqual(run_worker(args), 0)

            trace_path = output_dir / "data" / "llm_reasoning_traces.jsonl"
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 10)
            self.assertEqual(
                [record["action"] for record in records],
                ["purchase", "production"] * 5,
            )
            self.assertTrue(
                all(
                    record["reasoning_trace"] == "synthetic reasoning"
                    for record in records
                )
            )
            self.assertTrue(all(record["phase"] == "performance" for record in records))

    def test_llm_supplier_empty_responses_fall_back_to_zero_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_dir = write_contract(root / "contract-1", environment_id=1)
            output_dir = root / "output"
            runtime = SimpleNamespace(
                agents=(EmptySupplierAgent(),),
                models=(FakeGeminiModel(),),
                configs=(ProviderConfig("gemini", "gemini-test", "high"),),
            )
            args = Namespace(
                env=1,
                contract_dir=contract_dir,
                customer="rcc",
                supplier="gemini",
                seed=42,
                output_dir=output_dir,
                goal_treatment="control",
            )

            with mock.patch(
                "performance.orchestration.worker.build_llm_runtime",
                return_value=runtime,
            ):
                self.assertEqual(run_worker(args), 0)

            summary = json.loads(
                (output_dir / "data" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["valid"])
            for decision in summary["production_history"]:
                self.assertEqual(decision["ordered"], {"X": 0, "Y": 0, "Z": 0})
                self.assertEqual(decision["produced"], {"A": 0, "B": 0, "C": 0})
                self.assertEqual(
                    decision["supplier_purchase_decision"],
                    {"fallback": "empty_model_response"},
                )
                self.assertEqual(
                    decision["supplier_production_decision"],
                    {"fallback": "empty_model_response"},
                )


if __name__ == "__main__":
    unittest.main()
