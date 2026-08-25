"""Provider clients and phase-neutral append-only chat adapters."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from concordia.language_model import language_model

from ..prompting import format_system_prompt
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    ProviderConfig,
    RuntimeOptions,
    resolve_api_key,
)


API_MAX_RETRIES = 5
API_BASE_DELAY_SECONDS = 2.0
GEMINI_REQUEST_TIMEOUT_MS = 300_000


def retry_api(
    function: Callable[..., Any],
    *args: Any,
    max_retries: int = API_MAX_RETRIES,
    base_delay_seconds: float = API_BASE_DELAY_SECONDS,
    **kwargs: Any,
) -> Any:
    """Retry transient provider failures with exponential backoff and jitter."""
    for attempt in range(max_retries):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay_seconds * (2**attempt) + random.uniform(0, 1)
            print(
                f"  [API error: {exc!r} — retrying in {delay:.1f}s "
                f"({attempt + 1}/{max_retries})]"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    thought_trace: str | None
    prompt_cache_usage: dict[str, Any]
    diagnostics: dict[str, Any]


class EmptyModelResponseError(RuntimeError):
    """Raised after a provider returns no visible text twice for one turn."""

    def __init__(self, provider: str, model_name: str, diagnostics: dict[str, Any]):
        super().__init__(f"{provider}/{model_name} returned an empty response twice")
        self.provider = provider
        self.model_name = model_name
        self.diagnostics = diagnostics


class AppendOnlyChatLanguageModel(language_model.LanguageModel):
    """Provider-neutral chat history owned by exactly one agent."""

    def __init__(
        self,
        config: ProviderConfig,
        client: Any,
        options: RuntimeOptions = RuntimeOptions(),
        *,
        retry: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.provider = config.provider
        self.model_name = config.model
        self._client = client
        self._options = options
        self._retry = retry or retry_api
        self.system_prompt = ""
        self._messages: list[dict[str, str]] = []
        self.last_thought_trace: str | None = None
        self.last_prompt_cache_usage: dict[str, Any] | None = None
        self.last_response_diagnostics: dict[str, Any] | None = None
        self._agent_name: str | None = None

    def configure_agent(self, agent_name: str, info: str, goal: str) -> None:
        if self._agent_name is not None or self._messages:
            raise RuntimeError("A chat model instance can belong to only one agent")
        self._agent_name = str(agent_name)
        self.system_prompt = format_system_prompt(goal, info)

    @property
    def chat_metadata(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:16]
        return {
            "layout": "chat_v2",
            "message_count": len(self._messages),
            "system_prompt_sha256": digest,
        }

    def get_chat_state(self) -> dict[str, Any]:
        return {
            "version": "chat_v2",
            "system_prompt": self.system_prompt,
            "messages": [dict(message) for message in self._messages],
        }

    def set_chat_state(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != "chat_v2":
            raise ValueError("Unsupported chat-state version")
        messages = state.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Chat messages must be a list")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be an object")
            expected_role = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != expected_role:
                raise ValueError("Chat messages must alternate user/assistant roles")
            if not isinstance(message.get("content"), str):
                raise ValueError("Chat message content must be text")
        self.system_prompt = str(state.get("system_prompt", self.system_prompt))
        self._messages = [dict(message) for message in messages]

    def sample_chat(self, user_turn: str) -> str:
        user_turn = str(user_turn)
        candidate = [*self._messages, {"role": "user", "content": user_turn}]
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, 3):
            result = self._generate(candidate, attempt=attempt)
            is_empty = not bool(result.text.strip())
            attempts.append(
                {
                    "attempt": attempt,
                    "empty": is_empty,
                    "provider": result.diagnostics,
                    "prompt_cache": result.prompt_cache_usage,
                }
            )
            self.last_thought_trace = result.thought_trace
            self.last_prompt_cache_usage = result.prompt_cache_usage
            self.last_response_diagnostics = {
                **result.diagnostics,
                "generation_attempt": attempt,
                "empty_response_attempts": sum(item["empty"] for item in attempts),
            }
            text = result.text.strip()
            if text:
                if attempt > 1:
                    self.last_response_diagnostics["attempts"] = attempts
                self._messages.extend(
                    (
                        {"role": "user", "content": user_turn},
                        {"role": "assistant", "content": text},
                    )
                )
                return text
        self.last_response_diagnostics = {
            "reason": "empty_response_after_retry",
            "attempts": attempts,
        }
        raise EmptyModelResponseError(
            self.provider,
            self.model_name,
            self.last_response_diagnostics,
        )

    def sample_text(self, prompt: str, **_: Any) -> str:
        return self.sample_chat(prompt)

    def sample_choice(
        self,
        prompt: str,
        responses: list[str],
        **_: Any,
    ) -> tuple[int, str, dict[str, float]]:
        del prompt
        return 0, responses[0], {responses[0]: 1.0}

    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        attempt: int = 1,
    ) -> GenerationResult:
        del messages, attempt
        raise NotImplementedError


class GeminiLanguageModel(AppendOnlyChatLanguageModel):
    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        attempt: int = 1,
    ) -> GenerationResult:
        del attempt
        from google.genai import types

        contents = [
            types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=message["content"])],
            )
            for message in messages
        ]
        request_config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(
                thinking_level=self.config.reasoning,
                include_thoughts=self._options.thought_traces_enabled,
            ),
        )
        response = self._retry(
            self._client.models.generate_content,
            model=self.model_name,
            contents=contents,
            config=request_config,
        )
        thoughts: list[str] = []
        answers: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            for part in getattr(candidates[0].content, "parts", None) or []:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    thoughts.append(text)
                else:
                    answers.append(text)
        if not answers and getattr(response, "text", None):
            answers.append(response.text)
        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            text="\n".join(answers).strip(),
            thought_trace="\n".join(thoughts) if thoughts else None,
            prompt_cache_usage={
                "enabled": True,
                "requested": self._options.prompt_cache_enabled,
                "mode": "provider_implicit",
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "cached_tokens": getattr(usage, "cached_content_token_count", 0),
                "candidate_tokens": getattr(usage, "candidates_token_count", None),
                "thought_tokens": getattr(usage, "thoughts_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            },
            diagnostics={
                "candidate_count": len(candidates),
                "finish_reasons": [
                    str(getattr(candidate, "finish_reason", None))
                    for candidate in candidates
                ],
            },
        )


class OpenAILanguageModel(AppendOnlyChatLanguageModel):
    def __init__(
        self,
        config: ProviderConfig,
        client: Any,
        options: RuntimeOptions = RuntimeOptions(),
        *,
        retry: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(config, client, options, retry=retry)
        self._prompt_cache_key: str | None = None

    def configure_agent(self, agent_name: str, info: str, goal: str) -> None:
        super().configure_agent(agent_name, info, goal)
        identity = "\0".join(
            (
                self._options.cache_scope,
                self.model_name,
                str(agent_name),
                str(info),
                str(goal),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        self._prompt_cache_key = f"ratconbench:{digest}"

    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        attempt: int = 1,
    ) -> GenerationResult:
        del attempt
        reasoning = {"effort": self.config.reasoning}
        if self._options.thought_traces_enabled:
            reasoning["summary"] = "auto"
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "reasoning": reasoning,
            "input": [
                {"role": "system", "content": self.system_prompt},
                *[dict(message) for message in messages],
            ],
        }
        if self._options.prompt_cache_enabled:
            request_kwargs.update(
                prompt_cache_key=self._prompt_cache_key,
                prompt_cache_options={"mode": "implicit", "ttl": "30m"},
            )
        response = self._retry(self._client.responses.create, **request_kwargs)
        usage = getattr(response, "usage", None)
        details = getattr(usage, "input_tokens_details", None)
        output = getattr(response, "output", None) or []
        thought_trace = None
        if self._options.thought_traces_enabled:
            for block in output:
                if getattr(block, "type", None) != "reasoning":
                    continue
                summaries = [
                    item.text
                    for item in (getattr(block, "summary", None) or [])
                    if getattr(item, "text", None)
                ]
                thought_trace = "\n".join(summaries) if summaries else None
                break
        return GenerationResult(
            text=str(getattr(response, "output_text", "") or "").strip(),
            thought_trace=thought_trace,
            prompt_cache_usage={
                "enabled": self._options.prompt_cache_enabled,
                "mode": "implicit" if self._options.prompt_cache_enabled else None,
                "input_tokens": getattr(usage, "input_tokens", None),
                "cached_tokens": getattr(details, "cached_tokens", 0),
                "cache_write_tokens": getattr(details, "cache_write_tokens", 0),
            },
            diagnostics={
                "response_id": getattr(response, "id", None),
                "status": getattr(response, "status", None),
                "incomplete_details": _json_safe(
                    getattr(response, "incomplete_details", None)
                ),
                "output_types": [getattr(block, "type", None) for block in output],
                "output_tokens": getattr(usage, "output_tokens", None),
            },
        )


class AnthropicLanguageModel(AppendOnlyChatLanguageModel):
    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        attempt: int = 1,
    ) -> GenerationResult:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 16384,
            "system": self.system_prompt,
            "messages": [dict(message) for message in messages],
            "output_config": {"effort": self.config.reasoning},
            "thinking": {
                "type": "adaptive",
                "display": (
                    "summarized"
                    if self._options.thought_traces_enabled and attempt == 1
                    else "omitted"
                ),
            },
        }
        if self._options.prompt_cache_enabled:
            request_kwargs["cache_control"] = {"type": "ephemeral"}
        response = self._retry(self._client.messages.create, **request_kwargs)
        text_parts: list[str] = []
        thought_parts: list[str] = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text" and getattr(
                block, "text", None
            ):
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "thinking":
                thought = getattr(block, "thinking", None)
                if thought:
                    thought_parts.append(thought)
        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return GenerationResult(
            text="\n".join(text_parts).strip(),
            thought_trace="\n".join(thought_parts) if thought_parts else None,
            prompt_cache_usage={
                "enabled": self._options.prompt_cache_enabled,
                "mode": "automatic" if self._options.prompt_cache_enabled else None,
                "input_tokens": getattr(usage, "input_tokens", None),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    usage, "cache_read_input_tokens", 0
                ),
            },
            diagnostics={
                "response_id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "stop_details": _json_safe(getattr(response, "stop_details", None)),
                "content_types": [
                    getattr(block, "type", None)
                    for block in (getattr(response, "content", None) or [])
                ],
                "output_tokens": getattr(usage, "output_tokens", None),
                "thinking_tokens": getattr(output_details, "thinking_tokens", None),
            },
        )


class ClientRegistry:
    """Lazily create and reuse provider SDK clients within one worker."""

    def __init__(self, *, keys_dir: str | Path | None = None) -> None:
        self._keys_dir = keys_dir
        self._clients: dict[str, Any] = {}

    def get(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]
        api_key = resolve_api_key(provider, keys_dir=self._keys_dir)
        try:
            if provider == "gemini":
                import google.genai as genai
                from google.genai import types

                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(
                        timeout=GEMINI_REQUEST_TIMEOUT_MS,
                    ),
                )
            elif provider == "openai":
                import openai

                client = openai.OpenAI(api_key=api_key)
            elif provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"The {provider} SDK is not installed; install RatConBench's "
                "requirements."
            ) from exc
        self._clients[provider] = client
        return client


def build_chat_model(
    config: ProviderConfig,
    *,
    options: RuntimeOptions = RuntimeOptions(),
    client: Any | None = None,
    clients: ClientRegistry | None = None,
    retry: Callable[..., Any] | None = None,
) -> AppendOnlyChatLanguageModel:
    """Build one provider adapter without sharing its conversation state."""
    if client is None:
        client = (clients or ClientRegistry()).get(config.provider)
    model_types = {
        "gemini": GeminiLanguageModel,
        "openai": OpenAILanguageModel,
        "anthropic": AnthropicLanguageModel,
    }
    return model_types[config.provider](config, client, options, retry=retry)


def build_gemini_embedder(
    *,
    clients: ClientRegistry,
    model: str = DEFAULT_EMBEDDING_MODEL,
    retry: Callable[..., Any] | None = None,
) -> Callable[[str], np.ndarray]:
    """Build the legacy-compatible Gemini semantic-memory embedder."""
    client = clients.get("gemini")
    retry_call = retry or retry_api

    def embed(text: str) -> np.ndarray:
        response = retry_call(client.models.embed_content, model=model, contents=text)
        return np.array(response.embeddings[0].values, dtype=np.float32)

    return embed
