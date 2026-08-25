from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from concordia.typing import entity as entity_lib

from llm_utils.llm import (
    ProviderConfig,
    build_agent,
    load_agent_state,
    load_agent_states,
    save_agent_states,
)
from llm_utils.artifacts import PRIVATE_FILE_MODE


def _deterministic_embedder(_text: str) -> np.ndarray:
    return np.array([1.0, 0.0], dtype=np.float32)


class _CapturingModel:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.turns: list[str] = []
        self.messages: list[dict[str, str]] = []

    def configure_agent(self, name: str, info: str, goal: str) -> None:
        self.system_prompt = f"name={name}\ngoal={goal}\nprivate={info}"

    def sample_chat(self, turn: str) -> str:
        self.turns.append(turn)
        self.messages.extend(
            [
                {"role": "user", "content": turn},
                {"role": "assistant", "content": '{"response":"ok"}'},
            ]
        )
        return '{"response":"ok"}'

    def get_chat_state(self) -> dict:
        return {
            "version": "chat_v2",
            "system_prompt": self.system_prompt,
            "messages": list(self.messages),
        }

    def set_chat_state(self, state: dict) -> None:
        self.system_prompt = state["system_prompt"]
        self.messages = list(state["messages"])


class SharedAgentConstructionTests(unittest.TestCase):
    def test_agents_keep_private_prompts_and_chat_histories_separate(self) -> None:
        customer_model = _CapturingModel()
        caterer_model = _CapturingModel()
        customer = build_agent(
            "customer",
            customer_model,
            "CUSTOMER-ONLY",
            "customer goal",
            embedder=_deterministic_embedder,
        )
        build_agent(
            "caterer",
            caterer_model,
            "CATERER-ONLY",
            "caterer goal",
            embedder=_deterministic_embedder,
        )
        customer.act(
            entity_lib.ActionSpec(
                call_to_action="start",
                output_type=entity_lib.OutputType.FREE,
            )
        )

        self.assertIn("CUSTOMER-ONLY", customer_model.system_prompt)
        self.assertNotIn("CATERER-ONLY", customer_model.system_prompt)
        self.assertIn("CATERER-ONLY", caterer_model.system_prompt)
        self.assertEqual(len(customer_model.messages), 2)
        self.assertEqual(caterer_model.messages, [])

    def test_observations_enter_memory_and_only_external_new_items_are_sent(self) -> None:
        model = _CapturingModel()
        agent = build_agent(
            "customer",
            model,
            "private",
            "goal",
            embedder=_deterministic_embedder,
        )
        own = "customer: my previous public message"
        external = "caterer: shortfall rolls into the next round"
        agent.observe(own)
        agent.observe(external)
        agent.act(
            entity_lib.ActionSpec(
                call_to_action="What protection was discussed?",
                output_type=entity_lib.OutputType.FREE,
            )
        )

        turn = model.turns[0]
        new_observations = turn.split("Relevant past context:", 1)[0]
        self.assertIn(external, new_observations)
        self.assertNotIn(own, new_observations)
        self.assertIn("Relevant past context:", turn)

        memory = agent.get_state()["context_components"]["__memory__"]
        serialized = json.loads(memory["memory_bank"]["memory_bank"])
        stored = " ".join(serialized["text"].values())
        self.assertIn(own, stored)
        self.assertIn(external, stored)

    def test_agent_state_is_serialized_with_provider_identity(self) -> None:
        models = [_CapturingModel(), _CapturingModel()]
        agents = [
            build_agent(
                role,
                model,
                f"{role} info",
                f"{role} goal",
                embedder=_deterministic_embedder,
            )
            for role, model in zip(("customer", "caterer"), models, strict=True)
        ]
        configs = [
            ProviderConfig("openai", "gpt-test", "high"),
            ProviderConfig("anthropic", "claude-test", "high"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_agent_states(temp_dir, agents, configs)
            payloads = [json.loads(path.read_text()) for path in paths]
            modes = [stat.S_IMODE(path.stat().st_mode) for path in paths]

        self.assertEqual([item["provider"] for item in payloads], ["openai", "anthropic"])
        self.assertEqual([item["reasoning"] for item in payloads], ["high", "high"])
        self.assertTrue(all(item["state_version"] == 1 for item in payloads))
        self.assertTrue(all("state" in item for item in payloads))
        self.assertEqual(modes, [PRIVATE_FILE_MODE, PRIVATE_FILE_MODE])

    def test_agent_state_filenames_cannot_escape_or_collide(self) -> None:
        config = ProviderConfig("openai", "gpt-test", "high")

        def fake_agent(name: str) -> object:
            return type(
                "FakeAgent",
                (),
                {"name": name, "get_state": lambda self: {}},
            )()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "agent names"):
                save_agent_states(root / "run", [fake_agent("../escaped")], [config])
            self.assertFalse((root / "escaped_openai.json").exists())
            self.assertFalse((root / "run" / "agents").exists())

            with self.assertRaisesRegex(ValueError, "duplicate agent state"):
                save_agent_states(
                    root / "duplicate",
                    [fake_agent("customer"), fake_agent("customer")],
                    [config, config],
                )
            self.assertFalse((root / "duplicate" / "agents").exists())

            numeric_name = fake_agent("customer")
            numeric_name.name = 123
            with self.assertRaisesRegex(ValueError, "nonblank text"):
                save_agent_states(root / "numeric", [numeric_name], [config])
            self.assertFalse((root / "numeric" / "agents").exists())

            opaque_state = fake_agent("customer")
            opaque_state.get_state = lambda: {"opaque": object()}
            with self.assertRaisesRegex(TypeError, "Cannot serialize"):
                save_agent_states(root / "opaque", [opaque_state], [config])
            self.assertFalse((root / "opaque" / "agents").exists())

    def test_saved_pair_restores_memory_chat_and_provider_configuration(self) -> None:
        original_models = [_CapturingModel(), _CapturingModel()]
        original_agents = [
            build_agent(
                role,
                model,
                f"{role} private",
                f"{role} goal",
                embedder=_deterministic_embedder,
            )
            for role, model in zip(
                ("customer", "caterer"),
                original_models,
                strict=True,
            )
        ]
        original_agents[0].observe("caterer: proposed a rollover clause")
        original_agents[0].act(
            entity_lib.ActionSpec(
                call_to_action="Respond to the proposal",
                output_type=entity_lib.OutputType.FREE,
            )
        )
        original_states = [agent.get_state() for agent in original_agents]
        configs = [
            ProviderConfig("openai", "gpt-test", "medium"),
            ProviderConfig("anthropic", "claude-test", "high"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            save_agent_states(temp_dir, original_agents, configs)
            restored_models = [_CapturingModel(), _CapturingModel()]
            with mock.patch(
                "llm_utils.llm.agents.build_chat_model",
                side_effect=restored_models,
            ):
                restored = load_agent_states(
                    temp_dir,
                    embedder=_deterministic_embedder,
                    expected_roles={"customer", "caterer"},
                )

        self.assertEqual(set(restored), {"customer", "caterer"})
        self.assertEqual(
            restored["customer"].agent.get_state(),
            original_states[0],
        )
        self.assertEqual(
            restored["caterer"].agent.get_state(),
            original_states[1],
        )
        self.assertEqual(restored["customer"].provider_config, configs[0])
        self.assertEqual(restored["caterer"].provider_config, configs[1])
        self.assertEqual(
            restored["customer"].model.messages,
            original_models[0].messages,
        )

    def test_restore_supports_a_provider_override(self) -> None:
        agent = build_agent(
            "customer",
            _CapturingModel(),
            "private",
            "goal",
            embedder=_deterministic_embedder,
        )
        saved_config = ProviderConfig("openai", "gpt-test", "high")
        override = ProviderConfig("anthropic", "claude-override", "medium")
        restored_model = _CapturingModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_agent_states(temp_dir, [agent], [saved_config])[0]
            with mock.patch(
                "llm_utils.llm.agents.build_chat_model",
                return_value=restored_model,
            ):
                restored = load_agent_state(
                    path,
                    embedder=_deterministic_embedder,
                    provider_override=override,
                )
                collection = load_agent_states(
                    temp_dir,
                    embedder=_deterministic_embedder,
                    provider_overrides={"customer": override},
                    expected_roles={"customer"},
                )
                with self.assertRaisesRegex(ValueError, "missing caterer"):
                    load_agent_states(
                        temp_dir,
                        embedder=_deterministic_embedder,
                        expected_roles={"customer", "caterer"},
                    )

        self.assertEqual(restored.role, "customer")
        self.assertIs(restored.model, restored_model)
        self.assertEqual(restored.provider_config, override)
        self.assertEqual(restored.agent.get_state(), agent.get_state())
        self.assertEqual(set(collection), {"customer"})


if __name__ == "__main__":
    unittest.main()
