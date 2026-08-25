from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from llm_utils.llm.config import ProviderConfig, RuntimeOptions
from llm_utils.llm.factory import LLMRuntimeBundle, build_llm_runtime


def _provided_embedder(_text: str) -> np.ndarray:
    return np.array([1.0], dtype=np.float32)


class LLMRuntimeFactoryTests(unittest.TestCase):
    def test_builds_ordered_runtime_with_one_registry_and_embedder(self) -> None:
        prompts = [
            {"name": "buyer", "info": "buyer info", "goal": "buyer goal"},
            {"name": "seller", "info": "seller info", "goal": "seller goal"},
        ]
        provider_names = ["openai", "anthropic"]
        configs = (
            ProviderConfig("openai", "gpt-test", "high"),
            ProviderConfig("anthropic", "claude-test", "medium"),
        )
        models = (mock.sentinel.buyer_model, mock.sentinel.seller_model)
        agents = (mock.sentinel.buyer_agent, mock.sentinel.seller_agent)
        registry = mock.sentinel.registry
        generated_embedder = mock.Mock(name="generated_embedder")
        options = RuntimeOptions(cache_scope="factory-test")

        with (
            mock.patch(
                "llm_utils.llm.factory.load_provider_config",
                side_effect=configs,
            ) as load_config,
            mock.patch(
                "llm_utils.llm.factory.ClientRegistry",
                return_value=registry,
            ) as registry_type,
            mock.patch(
                "llm_utils.llm.factory.build_chat_model",
                side_effect=models,
            ) as build_model,
            mock.patch(
                "llm_utils.llm.factory.build_gemini_embedder",
                return_value=generated_embedder,
            ) as build_embedder,
            mock.patch(
                "llm_utils.llm.factory.build_agent",
                side_effect=agents,
            ) as build_runtime_agent,
        ):
            bundle = build_llm_runtime(
                prompts,
                provider_names,
                options=options,
            )

        self.assertIsInstance(bundle, LLMRuntimeBundle)
        self.assertEqual(bundle.configs, configs)
        self.assertEqual(bundle.models, models)
        self.assertEqual(bundle.agents, agents)
        self.assertIs(bundle.clients, registry)
        self.assertIs(bundle.embedder, generated_embedder)
        load_config.assert_has_calls([mock.call("openai"), mock.call("anthropic")])
        registry_type.assert_called_once_with()
        self.assertEqual(
            build_model.call_args_list,
            [
                mock.call(configs[0], options=options, clients=registry),
                mock.call(configs[1], options=options, clients=registry),
            ],
        )
        build_embedder.assert_called_once_with(clients=registry)
        self.assertEqual(
            build_runtime_agent.call_args_list,
            [
                mock.call(
                    "buyer",
                    models[0],
                    "buyer info",
                    "buyer goal",
                    embedder=generated_embedder,
                ),
                mock.call(
                    "seller",
                    models[1],
                    "seller info",
                    "seller goal",
                    embedder=generated_embedder,
                ),
            ],
        )

    def test_reuses_optional_clients_and_embedder(self) -> None:
        prompt = {"name": "observer", "info": "private", "goal": "inspect"}
        config = ProviderConfig("gemini", "gemini-test", "high")
        registry = mock.Mock(name="provided_registry")
        model = mock.sentinel.model
        agent = mock.sentinel.agent

        with (
            mock.patch(
                "llm_utils.llm.factory.load_provider_config",
                return_value=config,
            ),
            mock.patch("llm_utils.llm.factory.ClientRegistry") as registry_type,
            mock.patch(
                "llm_utils.llm.factory.build_chat_model",
                return_value=model,
            ) as build_model,
            mock.patch(
                "llm_utils.llm.factory.build_gemini_embedder"
            ) as build_embedder,
            mock.patch(
                "llm_utils.llm.factory.build_agent",
                return_value=agent,
            ) as build_runtime_agent,
        ):
            bundle = build_llm_runtime(
                [prompt],
                ["gemini"],
                clients=registry,
                embedder=_provided_embedder,
            )

        registry_type.assert_not_called()
        build_embedder.assert_not_called()
        build_model.assert_called_once_with(
            config,
            options=RuntimeOptions(),
            clients=registry,
        )
        build_runtime_agent.assert_called_once_with(
            "observer",
            model,
            "private",
            "inspect",
            embedder=_provided_embedder,
        )
        self.assertIs(bundle.clients, registry)
        self.assertIs(bundle.embedder, _provided_embedder)

    def test_cardinality_is_validated_before_runtime_construction(self) -> None:
        prompts = [
            {"name": "one", "info": "info", "goal": "goal"},
            {"name": "two", "info": "info", "goal": "goal"},
        ]

        with (
            mock.patch("llm_utils.llm.factory.load_provider_config") as load_config,
            mock.patch("llm_utils.llm.factory.ClientRegistry") as registry_type,
            self.assertRaisesRegex(ValueError, "equal lengths"),
        ):
            build_llm_runtime(prompts, ["openai"])

        load_config.assert_not_called()
        registry_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
