from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _support import BUNDLE_ROOT, PERFORMANCE_ROOT, load_module
from llm_utils.llm import ProviderConfig
from llm_utils.process_runner import ProcessJob
from performance.contracts import contract_id
from performance.tests.helpers import write_contract


domain_runtime = load_module(
    "domain_robustness_llm_runtime",
    BUNDLE_ROOT / "llm_runtime.py",
)
performance_run = load_module(
    "domain_robustness_performance_run",
    PERFORMANCE_ROOT / "run.py",
)


class FakeAgent:
    name = "hotel owner"

    def get_state(self):
        return {"context_components": {}}


class ScriptedCustomerAgent(FakeAgent):
    name = "customer"

    def __init__(self) -> None:
        self.observations = []

    def act(self, _action_spec):
        return '{"payment":10}'

    def observe(self, observation: str) -> None:
        self.observations.append(observation)


class ScriptedSupplierAgent(ScriptedCustomerAgent):
    name = "cleaning provider"

    def act(self, action_spec):
        if "buy_Detergent" in action_spec.call_to_action:
            return '{"buy_Detergent":0,"buy_Disinfectant":0,"buy_Wet_Wipes":0}'
        return (
            '{"perform_Conference_Room_Cleaning":0,'
            '"perform_King_Sized_Room_Cleaning":0,'
            '"perform_Single_Room_Cleaning":0}'
        )


class RuntimeAndLifecycleParityTests(unittest.TestCase):
    def test_runtime_delegates_to_shared_builder_with_outside_options(self) -> None:
        model = SimpleNamespace(system_prompt="")
        bundle = SimpleNamespace(models=(model,))
        prompts = [{"name": "customer", "info": "info", "goal": "goal"}]
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"PROMPT_CACHE": "false"},
        ), mock.patch.object(
            domain_runtime,
            "build_llm_runtime",
            return_value=bundle,
        ) as shared:
            options = domain_runtime.runtime_options(Path(temporary))
            result = domain_runtime.build_runtime(
                prompts,
                ["openai"],
                output_dir=Path(temporary),
                log_reasoning=False,
                formatters=[lambda goal, info: f"{goal}|{info}"],
                options=options,
            )

        self.assertIs(result, bundle)
        options = shared.call_args.kwargs["options"]
        self.assertFalse(options.prompt_cache_enabled)
        self.assertTrue(options.thought_traces_enabled)
        self.assertEqual(model.system_prompt, "goal|info")

    def test_domain_state_file_uses_shared_payload_schema(self) -> None:
        config = ProviderConfig("openai", "model-test", "high")
        with tempfile.TemporaryDirectory() as temporary:
            saved = domain_runtime.save_domain_agent_states(
                temporary,
                [FakeAgent()],
                [config],
                roles=["customer"],
                slugs=["hotel_owner"],
            )
            payload = json.loads(saved[0].read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload),
            {"state_version", "role", "provider", "model", "reasoning", "agent_name", "state"},
        )
        self.assertEqual(payload["role"], "customer")
        self.assertEqual(payload["agent_name"], "hotel owner")

    def test_resume_requires_complete_status_and_valid_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            (output / "data").mkdir(parents=True)
            job = ProcessJob(
                label="test",
                argv=("python",),
                output_dir=output,
                log_path=output / "stdout.log",
                cwd=Path(temporary),
            )
            (output / "data" / "summary.json").write_text('{"valid":true}')
            self.assertFalse(performance_run._completed_for_resume(job))
            (output / "run_status.json").write_text('{"status":"complete"}')
            self.assertTrue(performance_run._completed_for_resume(job))
            (output / "data" / "summary.json").write_text('{"valid":false}')
            self.assertFalse(performance_run._completed_for_resume(job))

    def test_domain_performance_worker_writes_outside_lifecycle_artifacts(self) -> None:
        worker_module = sys.modules[performance_run.run_one.__module__]
        side_module = sys.modules[worker_module.LLMSide.__module__]
        agent = ScriptedSupplierAgent()
        model = SimpleNamespace(system_prompt="")
        runtime = SimpleNamespace(
            agents=(agent,),
            models=(model,),
            configs=(ProviderConfig("openai", "model-test", "high"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_dir = write_contract(root / "contract", environment_id=5)
            (contract_dir / "contract.txt").write_text(
                "Parties: Customer and Caterer.\n"
                "Margherita Pizza: $10\n"
                "Week 1 payment: $10",
                encoding="utf-8",
            )
            output = root / "output"
            with mock.patch.object(side_module, "build_runtime", return_value=runtime):
                summary = worker_module.run_one(
                    domain="hotel-cleaning",
                    role="cleaning-provider",
                    contracts_root=root,
                    contract_id=contract_id(contract_dir),
                    provider="openai",
                    baseline_name="rc",
                    seed=42,
                    out_dir=output,
                    log_reasoning=False,
                    goal_treatment="control",
                )
            status = json.loads((output / "run_status.json").read_text())
            saved = json.loads(
                (output / "agents" / "cleaning_provider_openai.json").read_text()
            )

        self.assertTrue(summary["valid"])
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["valid"])
        self.assertEqual(saved["model"], "model-test")
        self.assertEqual(saved["role"], "supplier")


if __name__ == "__main__":
    unittest.main()
