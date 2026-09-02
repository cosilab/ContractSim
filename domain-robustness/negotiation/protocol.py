"""Domain-language adapter over the repository negotiation protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from concordia.typing import entity as entity_lib

from negotiation.logic.negotiation import run_negotiation as run_shared_negotiation


_LANGUAGE = {
    "hotel-cleaning": {
        "customer": "hotel owner",
        "Customer": "Hotel Owner",
        "caterer": "cleaning provider",
        "Caterer": "Cleaning Provider",
        "delivery and payment": "cleaning-service and payment",
    },
    "ai-model-hosting": {
        "customer": "AI startup founder",
        "Customer": "AI Startup Founder",
        "caterer": "AI service provider",
        "Caterer": "AI Service Provider",
        "delivery and payment": "service and payment",
    },
}


def translate_protocol_text(domain: str, text: str) -> str:
    """Translate protocol-owned canonical vocabulary into one domain."""
    language = _LANGUAGE[domain]
    translated = str(text).replace(
        "delivery and payment",
        language["delivery and payment"],
    )
    # Replace capitalized forms first so the lowercase replacement cannot alter
    # their domain-specific capitalization.
    for source in ("Customer", "Caterer", "customer", "caterer"):
        translated = translated.replace(source, language[source])
    return translated


def _replace_name_prefix(text: str, source: str, target: str) -> str:
    for suffix in (":", " "):
        prefix = f"{source}{suffix}"
        if text.lower().startswith(prefix.lower()):
            return f"{target}{suffix}{text[len(prefix):]}"
    return text


class _DomainAgent:
    """Present canonical roles to the protocol and domain roles to the model."""

    def __init__(
        self,
        domain: str,
        canonical_name: str,
        agent: Any,
        model: Any,
        *,
        debug_json: bool,
    ) -> None:
        self.domain = domain
        self.name = canonical_name
        self.agent = agent
        self.model = model
        self.debug_json = debug_json
        self.last_raw_output = ""
        self.last_call_to_action = ""

    def act(self, action_spec: entity_lib.ActionSpec) -> str:
        self.last_call_to_action = translate_protocol_text(
            self.domain,
            action_spec.call_to_action,
        )
        translated_spec = entity_lib.ActionSpec(
            call_to_action=self.last_call_to_action,
            output_type=action_spec.output_type,
            options=action_spec.options,
            tag=action_spec.tag,
        )
        self.last_raw_output = self.agent.act(translated_spec)
        if self.debug_json:
            trace = getattr(self.model, "last_thought_trace", None)
            if trace:
                print(f"\n[THOUGHT TRACE {self.agent.name}]:\n{trace}")
            print(f"\n[DEBUG JSON {self.agent.name}]: {self.last_raw_output}")
        return _replace_name_prefix(
            self.last_raw_output,
            self.agent.name,
            self.name,
        )

    def observe(self, observation: str) -> None:
        self.agent.observe(translate_protocol_text(self.domain, observation))


def run_negotiation(
    domain: str,
    agents: Sequence[Any],
    models: Sequence[Any],
    *,
    max_rounds: int,
    log_path: Path,
    contract_path: Path,
    debug_json: bool,
    record_llm_call: Callable[..., None],
) -> tuple[str | None, bool, dict[str, Any]]:
    """Run the outside protocol while translating only model-facing language."""
    if len(agents) != 2 or len(models) != 2:
        raise ValueError("negotiation requires exactly two agents and two models")
    adapted = (
        _DomainAgent(
            domain,
            "customer",
            agents[0],
            models[0],
            debug_json=debug_json,
        ),
        _DomainAgent(
            domain,
            "caterer",
            agents[1],
            models[1],
            debug_json=debug_json,
        ),
    )
    by_canonical_name = {agent.name: agent for agent in adapted}

    def record(canonical_name: str, model: Any, **payload: Any) -> None:
        adapter = by_canonical_name[canonical_name]
        context = dict(payload.get("context") or {})
        if "call_to_action" in context:
            context["call_to_action"] = translate_protocol_text(
                domain,
                context["call_to_action"],
            )
        payload["context"] = context
        if adapter.last_raw_output:
            payload["raw_output"] = adapter.last_raw_output
        record_llm_call(adapter.agent.name, model, **payload)

    contract, accepted, status = run_shared_negotiation(
        adapted,
        models,
        max_rounds=max_rounds,
        log_path=log_path,
        contract_path=contract_path,
        record_llm_call=record,
    )
    status = dict(status)
    invalid_agent = status.get("invalid_agent")
    if invalid_agent in by_canonical_name:
        status["invalid_agent"] = by_canonical_name[invalid_agent].agent.name
    return contract, accepted, status
