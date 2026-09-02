"""Minimal LLM and canonical-baseline sides for a performance match."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from domain import format_money
from actions import Actions
from prompts import build as build_prompts, formatter
from llm_runtime import build_runtime
from llm_utils.artifacts import write_json
from llm_utils.llm import EmptyModelResponseError, TraceRecorder
from performance.baselines.customer import CustomerRC, CustomerRCC, CustomerRE
from performance.baselines.supplier import SupplierRC, SupplierRCC, SupplierRE


CUSTOMER_BASELINES = {"rc": CustomerRC, "re": CustomerRE, "rcc": CustomerRCC}
SUPPLIER_BASELINES = {"rc": SupplierRC, "re": SupplierRE, "rcc": SupplierRCC}


def _observe(side: Any, observation: str) -> None:
    side._agent.observe(str(observation))


class LLMSide:
    def __init__(
        self,
        *,
        domain: str,
        canonical_role: str,
        provider: str,
        out_dir: Path,
        log_reasoning: bool,
        contract_text: str,
        goal_treatment: str = "control",
    ) -> None:
        self.domain = domain
        self.canonical_role = canonical_role
        self.total_paid = 0.0
        self.delivered = {"A": 0, "B": 0, "C": 0}
        role_index = 0 if canonical_role == "customer" else 1
        players = build_prompts(domain, contract_text, goal_treatment, canonical_role)
        self._out_dir = out_dir
        self._players = players
        prompt = players[role_index]
        system_formatter = formatter(domain)
        self.runtime = build_runtime(
            [prompt], [provider], output_dir=out_dir,
            log_reasoning=log_reasoning,
            formatters=[system_formatter],
        )
        self._model = self.runtime.models[0]
        self._agent = self.runtime.agents[0]
        self._recorder = TraceRecorder(
            out_dir / "data" / "llm_reasoning_traces.jsonl", enabled=log_reasoning
        )
        self.actions = Actions(domain, self._recorder.record)

    def prepare_artifacts(self) -> None:
        """Initialize worker-owned artifacts after its output claim succeeds."""
        write_json(
            self._out_dir / "config" / "performance_players.json",
            self._players,
        )
        self._recorder.reset()

    def decide_payment(self, *, week: int, remaining_budget: float):
        game = SimpleNamespace(total_weeks=11, customer_budget=remaining_budget)
        payment = self.actions.decide_payment(
            game, week, self._agent, self.total_paid, self._model
        )
        self.total_paid += payment
        message = (
            f"Week {week}: You paid {format_money(self.domain, payment)}. "
            f"Budget remaining: {format_money(self.domain, remaining_budget - payment)}."
        )
        _observe(self, message)
        return payment, None

    def observe_payment(self, *, week: int, payment: float):
        payer = {
            "hotel-cleaning": "Hotel Owner",
            "ai-model-hosting": "AI Startup Founder",
        }[self.domain]
        amount = format_money(self.domain, payment)
        _observe(self, f"Week {week}: {payer} paid {amount}.")
        return None

    def decide_purchases(self, game: Any, week: int):
        try:
            purchase = self.actions.decide_purchases(game, week, self._agent, self._model)
        except EmptyModelResponseError:
            return {"X": 0, "Y": 0, "Z": 0}, {"fallback": "empty_model_response"}
        return purchase, None

    def decide_production(
        self, game: Any, week: int, *, received: dict[str, int]
    ):
        try:
            production = self.actions.decide_production(
                game, week, self._agent, self._model, received=received
            )
        except EmptyModelResponseError:
            return {"A": 0, "B": 0, "C": 0}, {"fallback": "empty_model_response"}
        return production, None

    def observe_production(self, *, week: int, produced: dict[str, int]):
        for service, quantity in produced.items():
            self.delivered[service] += quantity
        if self.domain == "hotel-cleaning":
            message = (
                f"Week {week}: performed "
                f"{self.actions.format_services(produced, names=True)}; "
                f"cumulative cleaning services "
                f"{self.actions.format_services(self.delivered, names=True)}."
            )
        else:
            message = (
                f"Week {week}: provided "
                f"{self.actions.format_services(produced, names=True)}; "
                f"cumulative services "
                f"{self.actions.format_services(self.delivered, names=True)}."
            )
        _observe(self, message)
        return None

    def summary(self):
        if self.canonical_role == "customer":
            return {"type": "llm", "total_paid": self.total_paid}
        return {"type": "llm"}
