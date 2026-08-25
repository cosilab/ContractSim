from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from llm_utils.llm import (
    AnthropicLanguageModel,
    AppendOnlyChatLanguageModel,
    ClientRegistry,
    EmptyModelResponseError,
    GeminiLanguageModel,
    GenerationResult,
    OpenAILanguageModel,
    ProviderConfig,
    RuntimeOptions,
)


class _ScriptedModel(AppendOnlyChatLanguageModel):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__(
            ProviderConfig("gemini", "fake-model", "high"),
            object(),
        )
        self._outputs = iter(outputs)
        self.calls = 0

    def _generate(self, messages, *, attempt=1):
        self.calls += 1
        return GenerationResult(
            text=next(self._outputs),
            thought_trace=None,
            prompt_cache_usage={"mode": "fake"},
            diagnostics={"message_count": len(messages), "attempt": attempt},
        )


class SharedProviderAdapterTests(unittest.TestCase):
    def test_gemini_client_has_a_request_timeout(self) -> None:
        with (
            mock.patch(
                "llm_utils.llm.providers.resolve_api_key",
                return_value="test-key",
            ),
            mock.patch("google.genai.Client") as client_type,
        ):
            ClientRegistry().get("gemini")

        options = client_type.call_args.kwargs["http_options"]
        self.assertEqual(options.timeout, 300_000)

    def test_empty_response_retries_once_without_mutating_history(self) -> None:
        recovers = _ScriptedModel(["", '{"response":"ok"}'])
        recovers.configure_agent("customer", "secret", "goal")
        self.assertEqual(recovers.sample_chat("one turn"), '{"response":"ok"}')
        self.assertEqual(recovers.calls, 2)
        self.assertEqual(len(recovers.get_chat_state()["messages"]), 2)

        fails = _ScriptedModel([" ", ""])
        fails.configure_agent("customer", "secret", "goal")
        with self.assertRaises(EmptyModelResponseError):
            fails.sample_chat("one turn")
        self.assertEqual(fails.get_chat_state()["messages"], [])

    def test_openai_preserves_exact_append_only_prefix_and_cache_key(self) -> None:
        calls = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    id=f"response-{len(calls)}",
                    status="completed",
                    incomplete_details=None,
                    output=[],
                    output_text='{"response":"ok"}',
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=10,
                        input_tokens_details=SimpleNamespace(
                            cached_tokens=50,
                            cache_write_tokens=0,
                        ),
                    ),
                )

        model = OpenAILanguageModel(
            ProviderConfig("openai", "gpt-test", "high"),
            SimpleNamespace(responses=Responses()),
            RuntimeOptions(cache_scope="run-1", thought_traces_enabled=False),
        )
        model.configure_agent("customer", "secret", "goal")
        model.sample_chat("turn one")
        model.sample_chat("turn two")

        self.assertEqual(calls[0]["prompt_cache_key"], calls[1]["prompt_cache_key"])
        self.assertEqual(
            calls[1]["input"][:-1],
            calls[0]["input"]
            + [{"role": "assistant", "content": '{"response":"ok"}'}],
        )
        self.assertEqual(model.last_prompt_cache_usage["cached_tokens"], 50)

    def test_anthropic_uses_top_level_cache_and_hides_retry_thinking(self) -> None:
        calls = []

        class Messages:
            def create(self, **kwargs):
                calls.append(kwargs)
                content = (
                    [SimpleNamespace(type="thinking", thinking="internal")]
                    if len(calls) == 1
                    else [SimpleNamespace(type="text", text='{"response":"ok"}')]
                )
                return SimpleNamespace(
                    id=f"message-{len(calls)}",
                    stop_reason="end_turn",
                    stop_details=None,
                    content=content,
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=10,
                        output_tokens_details=None,
                        cache_creation_input_tokens=20,
                        cache_read_input_tokens=30,
                    ),
                )

        model = AnthropicLanguageModel(
            ProviderConfig("anthropic", "claude-test", "high"),
            SimpleNamespace(messages=Messages()),
        )
        model.configure_agent("caterer", "secret", "goal")
        self.assertEqual(model.sample_chat("turn"), '{"response":"ok"}')
        self.assertEqual(calls[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(calls[0]["thinking"]["display"], "summarized")
        self.assertEqual(calls[1]["thinking"]["display"], "omitted")
        self.assertEqual(calls[0]["messages"], calls[1]["messages"])

    def test_gemini_preserves_native_chat_roles(self) -> None:
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    text='{"response":"ok"}',
                    candidates=[
                        SimpleNamespace(
                            finish_reason="STOP",
                            content=SimpleNamespace(
                                parts=[
                                    SimpleNamespace(
                                        text='{"response":"ok"}',
                                        thought=False,
                                    )
                                ]
                            ),
                        )
                    ],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=100,
                        cached_content_token_count=50,
                        candidates_token_count=10,
                        thoughts_token_count=0,
                        total_token_count=110,
                    ),
                )

        model = GeminiLanguageModel(
            ProviderConfig("gemini", "gemini-test", "high"),
            SimpleNamespace(models=Models()),
            RuntimeOptions(thought_traces_enabled=False),
        )
        model.configure_agent("customer", "secret", "goal")
        model.sample_chat("turn one")
        model.sample_chat("turn two")

        self.assertEqual(
            [content.role for content in calls[1]["contents"]],
            ["user", "model", "user"],
        )
        self.assertEqual(
            calls[1]["config"].system_instruction,
            model.system_prompt,
        )


if __name__ == "__main__":
    unittest.main()
