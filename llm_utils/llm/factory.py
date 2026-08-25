"""Phase-neutral assembly of provider models and Concordia agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from concordia.agents import entity_agent

from .agents import build_agent
from .config import ProviderConfig, RuntimeOptions, load_provider_config
from .providers import (
    AppendOnlyChatLanguageModel,
    ClientRegistry,
    build_chat_model,
    build_gemini_embedder,
)


Embedder = Callable[[str], np.ndarray]

__all__ = ("LLMRuntimeBundle", "build_llm_runtime")


@dataclass(frozen=True)
class LLMRuntimeBundle:
    """One ordered set of isolated models, agents, and shared dependencies."""

    configs: tuple[ProviderConfig, ...]
    models: tuple[AppendOnlyChatLanguageModel, ...]
    agents: tuple[entity_agent.EntityAgent, ...]
    clients: ClientRegistry
    embedder: Embedder


def build_llm_runtime(
    role_prompts: Sequence[Mapping[str, str]],
    provider_names: Sequence[str],
    *,
    options: RuntimeOptions = RuntimeOptions(),
    clients: ClientRegistry | None = None,
    embedder: Embedder | None = None,
) -> LLMRuntimeBundle:
    """Build agents in the shared order of role prompts and provider names.

    Each role prompt supplies ``name``, ``info``, and ``goal``. The role name is
    also the resulting Concordia agent name. A single client registry and a
    single embedder are shared as infrastructure; every agent still receives a
    distinct append-only chat model.
    """
    if len(role_prompts) != len(provider_names):
        raise ValueError("role_prompts and provider_names must have equal lengths")

    # Resolve all caller-owned input before creating any partial runtime.
    prompts = tuple(
        (prompt["name"], prompt["info"], prompt["goal"])
        for prompt in role_prompts
    )
    providers = tuple(provider_names)

    configs = tuple(load_provider_config(provider) for provider in providers)
    registry = clients if clients is not None else ClientRegistry()
    models = tuple(
        build_chat_model(config, options=options, clients=registry)
        for config in configs
    )
    shared_embedder = (
        embedder
        if embedder is not None
        else build_gemini_embedder(clients=registry)
    )
    agents = tuple(
        build_agent(
            name,
            model,
            info,
            goal,
            embedder=shared_embedder,
        )
        for (name, info, goal), model in zip(prompts, models, strict=True)
    )
    return LLMRuntimeBundle(
        configs=configs,
        models=models,
        agents=agents,
        clients=registry,
        embedder=shared_embedder,
    )
