"""Domain-only adapters around the repository's shared LLM runtime."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from llm_utils.llm import (
    AGENT_STATE_VERSION,
    LLMRuntimeBundle,
    ProviderConfig,
    RuntimeOptions,
    TraceRecorder,
    build_llm_runtime,
)
from llm_utils.artifacts import json_safe, write_json


def _environment_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


def runtime_options(output_dir: str | Path) -> RuntimeOptions:
    """Return the outside worker's provider-affecting runtime options."""
    return RuntimeOptions(
        prompt_cache_enabled=_environment_flag("PROMPT_CACHE", True),
        thought_traces_enabled=True,
        cache_scope=str(Path(output_dir).resolve()),
    )


def build_runtime(
    prompts: Sequence[dict[str, str]],
    providers: Sequence[str],
    *,
    output_dir: Path,
    log_reasoning: bool,
    formatters: Sequence[Callable[[str, str], str] | None] | None = None,
    options: RuntimeOptions | None = None,
) -> LLMRuntimeBundle:
    """Build the same runtime as the outside phases, plus prompt formatting.

    ``log_reasoning`` controls only whether domain runs persist traces. Provider
    thought traces remain enabled exactly as in the outside workers, so turning
    local logging off cannot change a model request or response.
    """
    del log_reasoning
    resolved_options = options or RuntimeOptions(
        prompt_cache_enabled=True,
        thought_traces_enabled=True,
        cache_scope=str(output_dir.resolve()),
    )
    runtime = build_llm_runtime(
        prompts,
        providers,
        options=resolved_options,
    )
    if formatters is not None:
        for prompt, model, formatter in zip(
            prompts,
            runtime.models,
            formatters,
            strict=True,
        ):
            if formatter is not None:
                model.system_prompt = formatter(prompt["goal"], prompt["info"])
    return runtime


def save_domain_agent_states(
    output_dir: str | Path,
    agents: Sequence[object],
    provider_configs: Sequence[ProviderConfig],
    *,
    roles: Sequence[str],
    slugs: Sequence[str],
) -> list[Path]:
    """Persist shared agent-state payloads using safe domain filename slugs.

    Some domain-facing agent names contain spaces, which the shared filename
    validator intentionally rejects. This keeps the shared payload schema and
    restoration semantics while making that presentation-only filename choice
    explicit inside domain robustness.
    """
    lengths = {
        len(agents),
        len(provider_configs),
        len(roles),
        len(slugs),
    }
    if len(lengths) != 1:
        raise ValueError("agents, provider_configs, roles, and slugs must align")
    agents_dir = Path(output_dir) / "agents"
    planned: list[tuple[Path, dict[str, object]]] = []
    seen: set[Path] = set()
    for agent, config, role, slug in zip(
        agents,
        provider_configs,
        roles,
        slugs,
        strict=True,
    ):
        if (
            not slug
            or slug in {".", ".."}
            or Path(slug).name != slug
            or any(not (char.isalnum() or char in {"-", "_"}) for char in slug)
        ):
            raise ValueError(f"invalid domain agent-state slug: {slug!r}")
        path = agents_dir / f"{slug}_{config.provider}.json"
        if path in seen:
            raise ValueError(f"duplicate agent state output path: {path.name}")
        seen.add(path)
        payload = json_safe({
            "state_version": AGENT_STATE_VERSION,
            "role": role,
            "provider": config.provider,
            "model": config.model,
            "reasoning": config.reasoning,
            "agent_name": agent.name,
            "state": agent.get_state(),
        })
        assert isinstance(payload, dict)
        planned.append((path, payload))

    agents_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for path, payload in planned:
        write_json(path, payload)
        saved.append(path)
        print(f">>> Agent saved: {path}")
    return saved


def trace_recorder(output_dir: Path, enabled: bool) -> TraceRecorder:
    return TraceRecorder(
        output_dir / "data" / "llm_reasoning_traces.jsonl",
        enabled=enabled,
    )
