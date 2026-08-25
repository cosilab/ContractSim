"""Concordia agent construction and state persistence."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import constant as constant_component
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from ..artifacts import json_safe, write_json
from .config import ProviderConfig, RuntimeOptions, load_provider_config
from .providers import ClientRegistry, build_chat_model


AGENT_STATE_VERSION = 1


@dataclass(frozen=True)
class RestoredAgent:
    """An agent reconstructed from a saved RatConBench state."""

    role: str
    agent: entity_agent.EntityAgent
    model: Any
    provider_config: ProviderConfig
    source_path: Path


class NewObservationsComponent(entity_component.ContextComponent):
    """Buffer observations received since this agent's previous turn."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: list[str] = []
        self._lock = threading.Lock()

    def pre_observe(self, observation: str) -> str:
        with self._lock:
            self._pending.append(observation)
        return ""

    def pre_act(self, action_spec: Any) -> str:
        del action_spec
        with self._lock:
            observations = list(self._pending)
            self._pending.clear()
        return json.dumps(observations, ensure_ascii=False)

    def get_state(self) -> dict[str, list[str]]:
        with self._lock:
            return {"pending": list(self._pending)}

    def set_state(self, state: dict[str, Any]) -> None:
        pending = state.get("pending", [])
        if not isinstance(pending, list) or not all(
            isinstance(item, str) for item in pending
        ):
            raise ValueError("Pending observations must be text")
        with self._lock:
            self._pending = list(pending)


class AppendOnlyChatActComponent(entity_component.ActingComponent):
    """Turn Concordia context into exactly one provider chat turn."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def get_state(self) -> dict[str, Any]:
        return {"chat": self._model.get_chat_state()}

    def set_state(self, state: dict[str, Any]) -> None:
        if "chat" in state:
            self._model.set_chat_state(state["chat"])

    def get_action_attempt(self, context: dict[str, str], action_spec: Any) -> str:
        if action_spec.output_type not in entity_lib.FREE_ACTION_TYPES:
            raise NotImplementedError(
                "AppendOnlyChatActComponent supports free-text actions only"
            )
        name = self.get_entity().name
        try:
            pending = json.loads(context.get("new_observations", "[]"))
        except (json.JSONDecodeError, TypeError):
            pending = []
        own_prefix = f"{name}:".casefold()
        external_observations = [
            observation
            for observation in pending
            if not observation.lstrip().casefold().startswith(own_prefix)
        ]
        turn_parts: list[str] = []
        if external_observations:
            turn_parts.append(
                "Observations (ordered from oldest to latest):\n"
                + "\n".join(
                    f"[observation] {observation}"
                    for observation in external_observations
                )
            )
        recalled = context.get("semantic_recall", "").strip()
        if recalled:
            turn_parts.append(recalled)
        call_to_action = action_spec.call_to_action.replace("{name}", name)
        turn_parts.append(f"Exercise: {call_to_action}\nAnswer: {name} ")
        output = self._model.sample_chat("\n\n".join(turn_parts))
        return f"{name} {output}"


class ObservationToMemoryComponent(observation_component.ObservationToMemory):
    """Store observations without emitting an empty prompt label."""

    def pre_act(self, action_spec: Any) -> str:
        del action_spec
        return ""


class SemanticRecallComponent(entity_component.ContextComponent):
    def __init__(self, limit: int = 5) -> None:
        super().__init__()
        self._limit = limit

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        del state

    def pre_act(self, action_spec: Any) -> str:
        memory = self.get_entity().get_component(
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY
        )
        results = memory.retrieve_associative(
            action_spec.call_to_action,
            limit=self._limit,
        )
        if not results:
            return ""
        return "Relevant past context:\n" + "\n".join(
            f"  {result}" for result in results
        )


def build_agent(
    name: str,
    model: Any,
    info: str,
    goal: str,
    *,
    embedder: Callable[[str], np.ndarray],
    recall_limit: int = 5,
) -> entity_agent.EntityAgent:
    """Construct one isolated Concordia agent with private prompt and memory."""
    if not callable(embedder):
        raise TypeError("embedder must be callable")
    configure_agent = getattr(model, "configure_agent", None)
    if configure_agent is None:
        raise TypeError("chat models must implement configure_agent")
    configure_agent(name, info, goal)
    memory = memory_component.AssociativeMemory(
        memory_bank=basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )
    )
    return entity_agent.EntityAgent(
        agent_name=name,
        act_component=AppendOnlyChatActComponent(model=model),
        context_components={
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            "observation_to_memory": ObservationToMemoryComponent(),
            "new_observations": NewObservationsComponent(),
            "semantic_recall": SemanticRecallComponent(limit=recall_limit),
            "goal": constant_component.Constant(state=goal, pre_act_label="Goal"),
            "info": constant_component.Constant(
                state=info,
                pre_act_label="Private Info",
            ),
        },
    )


def save_agent_states(
    output_dir: str | Path,
    agents: Sequence[Any],
    provider_configs: Sequence[ProviderConfig],
) -> list[Path]:
    """Persist complete agent memory and provider-chat state."""
    if len(agents) != len(provider_configs):
        raise ValueError("agents and provider_configs must have equal lengths")
    agents_dir = Path(output_dir) / "agents"
    planned: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    for agent, config in zip(agents, provider_configs, strict=True):
        agent_name = agent.name
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent names used for state files must be nonblank text")
        role = agent_name.lower()
        if (
            not role
            or role in {".", ".."}
            or Path(role).name != role
            or any(
                not (character.isalnum() or character in {"-", "_"})
                for character in role
            )
        ):
            raise ValueError(
                "agent names used for state files may contain only letters, "
                "numbers, hyphens, and underscores"
            )
        payload = {
            "state_version": AGENT_STATE_VERSION,
            "role": role,
            "provider": config.provider,
            "model": config.model,
            "reasoning": config.reasoning,
            "agent_name": agent_name,
            "state": agent.get_state(),
        }
        path = agents_dir / f"{role}_{config.provider}.json"
        if path in seen_paths:
            raise ValueError(f"duplicate agent state output path: {path.name}")
        seen_paths.add(path)
        converted = json_safe(payload)
        assert isinstance(converted, dict)
        planned.append((path, converted))

    agents_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for path, payload in planned:
        write_json(path, payload)
        saved_paths.append(path)
        print(f">>> Agent saved: {path}")
    return saved_paths


def _read_saved_agent(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Saved agent does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved agent is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Saved agent must contain a JSON object: {path}")
    version = payload.get("state_version", AGENT_STATE_VERSION)
    if version != AGENT_STATE_VERSION:
        raise ValueError(
            f"Unsupported agent-state version {version!r} in {path}; "
            f"expected {AGENT_STATE_VERSION}"
        )
    return payload


def _saved_provider_config(payload: dict[str, Any], path: Path) -> ProviderConfig:
    try:
        provider = str(payload["provider"])
        model = str(payload["model"])
    except KeyError as exc:
        raise ValueError(f"Missing provider metadata in saved agent: {path}") from exc
    default = load_provider_config(provider)
    reasoning = str(payload.get("reasoning", default.reasoning))
    return ProviderConfig(provider=provider, model=model, reasoning=reasoning)


def _private_components(
    state: dict[str, Any],
    path: Path,
) -> tuple[str, str]:
    components = state.get("context_components")
    if not isinstance(components, dict):
        raise ValueError(f"Saved agent has no context components: {path}")
    info_component = components.get("info")
    goal_component = components.get("goal")
    if not isinstance(info_component, dict) or not isinstance(goal_component, dict):
        raise ValueError(f"Saved agent has no private prompt components: {path}")
    info = info_component.get("state")
    goal = goal_component.get("state")
    if not isinstance(info, str) or not isinstance(goal, str):
        raise ValueError(f"Saved agent prompt components must contain text: {path}")
    return info, goal


def load_agent_state(
    path: str | Path,
    *,
    embedder: Callable[[str], np.ndarray],
    options: RuntimeOptions = RuntimeOptions(),
    clients: ClientRegistry | None = None,
    provider_override: str | ProviderConfig | None = None,
) -> RestoredAgent:
    """Rebuild one agent and restore its memory and append-only chat history."""
    source_path = Path(path).expanduser().resolve()
    payload = _read_saved_agent(source_path)
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"Saved agent has no state object: {source_path}")
    agent_name = payload.get("agent_name")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError(f"Saved agent has no valid agent_name: {source_path}")
    role = str(payload.get("role", agent_name)).lower()

    if provider_override is None:
        config = _saved_provider_config(payload, source_path)
    elif isinstance(provider_override, ProviderConfig):
        config = provider_override
    else:
        config = load_provider_config(provider_override)

    registry = clients or ClientRegistry()
    model = build_chat_model(config, options=options, clients=registry)
    info, goal = _private_components(state, source_path)
    agent = build_agent(
        agent_name,
        model,
        info,
        goal,
        embedder=embedder,
    )
    agent.set_state(state)
    return RestoredAgent(
        role=role,
        agent=agent,
        model=model,
        provider_config=config,
        source_path=source_path,
    )


def load_agent_states(
    output_dir: str | Path,
    *,
    embedder: Callable[[str], np.ndarray],
    options: RuntimeOptions = RuntimeOptions(),
    clients: ClientRegistry | None = None,
    provider_overrides: dict[str, str | ProviderConfig] | None = None,
    expected_roles: Collection[str] | None = None,
) -> dict[str, RestoredAgent]:
    """Restore saved agents, optionally enforcing an exact set of roles."""
    agents_dir = Path(output_dir).expanduser().resolve() / "agents"
    paths = sorted(agents_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No saved agent states found in {agents_dir}")

    if isinstance(expected_roles, str):
        raise TypeError("expected_roles must be a collection of role names")
    required_roles = (
        {str(role).lower() for role in expected_roles}
        if expected_roles is not None
        else None
    )
    overrides = dict(provider_overrides or {})
    registry = clients or ClientRegistry()
    restored: dict[str, RestoredAgent] = {}
    for path in paths:
        payload = _read_saved_agent(path)
        role = str(payload.get("role", payload.get("agent_name", ""))).lower()
        if role in restored:
            raise ValueError(f"Duplicate saved state for role {role!r} in {agents_dir}")
        restored[role] = load_agent_state(
            path,
            embedder=embedder,
            options=options,
            clients=registry,
            provider_override=overrides.get(role),
        )

    unknown_overrides = sorted(set(overrides) - set(restored))
    if unknown_overrides:
        raise ValueError(
            "Provider overrides do not match saved roles: "
            + ", ".join(unknown_overrides)
        )
    if required_roles is not None and required_roles != set(restored):
        missing_roles = sorted(required_roles - set(restored))
        extra_roles = sorted(set(restored) - required_roles)
        details = []
        if missing_roles:
            details.append("missing " + ", ".join(missing_roles))
        if extra_roles:
            details.append("unexpected " + ", ".join(extra_roles))
        raise ValueError("Invalid saved agent set: " + "; ".join(details))
    return restored
