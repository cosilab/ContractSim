"""Execute and persist one performance matchup."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ORCHESTRATION_DIR = Path(__file__).resolve().parent
PERFORMANCE_DIR = ORCHESTRATION_DIR.parent
PROJECT_ROOT = PERFORMANCE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_utils import load_environment_metadata  # noqa: E402
from llm_utils.artifacts import claim_output_directory, write_json  # noqa: E402
from llm_utils.llm import (  # noqa: E402
    PROVIDERS,
    EmptyModelResponseError,
    RuntimeOptions,
    TraceRecorder,
    build_llm_runtime,
    save_agent_states,
)
from llm_utils.process_runner import (  # noqa: E402
    ProcessTermination,
    terminate_on_signals,
)
from performance.contracts import contract_id, validate_contract_environment  # noqa: E402
from performance.baselines.customer import (  # noqa: E402
    CustomerRC,
    CustomerRCC,
    CustomerRE,
)
from performance.baselines.supplier import (  # noqa: E402
    SupplierRC,
    SupplierRCC,
    SupplierRE,
)
from performance.engine import run_performance  # noqa: E402
from performance.logic import (  # noqa: E402
    customer_decide_payment,
    supplier_decide_production,
    supplier_decide_purchases,
)
from performance.logic.performance import DISH_NAMES  # noqa: E402
from performance.prompting import (  # noqa: E402
    GOAL_TREATMENT_CHOICES,
    build_prompts,
    treatment_metadata,
)


BASELINES = ("rc", "re", "rcc")
PARTICIPANTS = (*BASELINES, *PROVIDERS)
CUSTOMER_BASELINES = {
    "rc": CustomerRC,
    "re": CustomerRE,
    "rcc": CustomerRCC,
}
SUPPLIER_BASELINES = {
    "rc": SupplierRC,
    "re": SupplierRE,
    "rcc": SupplierRCC,
}


class _LLMCustomer:
    def __init__(
        self,
        agent: Any,
        model: Any,
        record_llm_call: Any,
        total_weeks: int,
    ) -> None:
        self.agent = agent
        self.model = model
        self.record_llm_call = record_llm_call
        self.total_weeks = total_weeks
        self.total_paid = 0.0
        self.delivered = {"A": 0, "B": 0, "C": 0}

    def decide_payment(
        self,
        *,
        week: int,
        remaining_budget: float,
    ) -> tuple[float, None]:
        game = SimpleNamespace(
            total_weeks=self.total_weeks,
            customer_budget=remaining_budget,
        )
        payment = customer_decide_payment(
            game,
            week,
            self.agent,
            self.total_paid,
            model=self.model,
            record_llm_call=self.record_llm_call,
        )
        self.total_paid += payment
        self.agent.observe(
            f"Week {week}: You paid ${payment:.2f}. "
            f"Budget remaining: ${remaining_budget - payment:.2f}."
        )
        return payment, None

    def observe_production(
        self,
        *,
        week: int,
        produced: dict[str, int],
    ) -> None:
        for dish, quantity in produced.items():
            self.delivered[dish] += quantity
        produced_text = ", ".join(
            f"{quantity} {DISH_NAMES[dish]}" for dish, quantity in produced.items()
        )
        delivered_text = ", ".join(
            f"{quantity} {DISH_NAMES[dish]}"
            for dish, quantity in self.delivered.items()
        )
        self.agent.observe(
            f"Week {week}: delivered {produced_text}; "
            f"cumulative deliveries {delivered_text}."
        )
        return None

    def summary(self) -> dict[str, Any]:
        return {"type": "llm", "total_paid": self.total_paid}


class _LLMSupplier:
    def __init__(self, agent: Any, model: Any, record_llm_call: Any) -> None:
        self.agent = agent
        self.model = model
        self.record_llm_call = record_llm_call

    def observe_payment(
        self,
        *,
        week: int,
        payment: float,
    ) -> None:
        message = f"Week {week}: Customer paid ${payment:.2f}."
        self.agent.observe(message)
        return None

    def decide_purchases(
        self,
        game: Any,
        week: int,
    ) -> tuple[dict[str, int], dict[str, str] | None]:
        try:
            purchase = supplier_decide_purchases(
                game,
                week,
                self.agent,
                model=self.model,
                record_llm_call=self.record_llm_call,
            )
        except EmptyModelResponseError:
            return {"X": 0, "Y": 0, "Z": 0}, {"fallback": "empty_model_response"}
        return purchase, None

    def decide_production(
        self,
        game: Any,
        week: int,
        *,
        received: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, str] | None]:
        try:
            production = supplier_decide_production(
                game,
                week,
                self.agent,
                received=received,
                model=self.model,
                record_llm_call=self.record_llm_call,
            )
        except EmptyModelResponseError:
            return {"A": 0, "B": 0, "C": 0}, {"fallback": "empty_model_response"}
        return production, None

    def summary(self) -> dict[str, str]:
        return {"type": "llm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--customer", choices=PARTICIPANTS, required=True)
    parser.add_argument("--supplier", choices=PARTICIPANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--goal-treatment",
        choices=GOAL_TREATMENT_CHOICES,
        default="control",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def run_worker(args: argparse.Namespace) -> int:
    contract_dir = args.contract_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_contract_environment(contract_dir, args.env)
    contract_path = contract_dir / "contract.txt"
    try:
        contract_text = contract_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot load {contract_path}: {exc}") from exc
    config = _read_json(contract_dir / "config.json")
    terms = _read_json(contract_dir / "terms.json")
    stable_contract_id = contract_id(contract_dir)
    environment = load_environment_metadata(args.env)

    llm_roles = [
        role
        for role, participant in (
            ("customer", args.customer),
            ("supplier", args.supplier),
        )
        if participant in PROVIDERS
    ]
    if args.goal_treatment != "control" and len(llm_roles) != 1:
        raise RuntimeError(
            "a named goal treatment requires exactly one LLM participant"
        )
    treated_role = llm_roles[0] if args.goal_treatment != "control" else None
    prompts = build_prompts(
        args.env,
        contract_text,
        goal_treatment=args.goal_treatment,
        treated_role=treated_role,
    )

    runtime = None
    agents_by_role: dict[str, Any] = {}
    models_by_role: dict[str, Any] = {}
    if llm_roles:
        prompt_by_role = {prompt["name"]: prompt for prompt in prompts}
        providers = {
            "customer": args.customer,
            "supplier": args.supplier,
        }
        runtime = build_llm_runtime(
            [prompt_by_role[role] for role in llm_roles],
            [providers[role] for role in llm_roles],
            options=RuntimeOptions(cache_scope=str(args.output_dir.resolve())),
        )
        agents_by_role = dict(zip(llm_roles, runtime.agents, strict=True))
        models_by_role = dict(zip(llm_roles, runtime.models, strict=True))

    trace_recorder = TraceRecorder(
        output_dir / "data" / "llm_reasoning_traces.jsonl",
        enabled=bool(llm_roles),
    )

    customer = (
        _LLMCustomer(
            agents_by_role["customer"],
            models_by_role["customer"],
            trace_recorder.record,
            total_weeks=11,
        )
        if args.customer in PROVIDERS
        else CUSTOMER_BASELINES[args.customer](
            config,
            terms,
            environment_id=args.env,
        )
    )
    supplier = (
        _LLMSupplier(
            agents_by_role["supplier"],
            models_by_role["supplier"],
            trace_recorder.record,
        )
        if args.supplier in PROVIDERS
        else SUPPLIER_BASELINES[args.supplier](
            stable_contract_id,
            contract_dir,
            config,
            terms,
            environment_id=args.env,
        )
    )

    claim = claim_output_directory(
        output_dir,
        metadata={
            "implementation": "Performance",
            "environment_id": args.env,
            "contract_id": stable_contract_id,
        },
    )
    with claim:
        trace_recorder.reset()
        write_json(output_dir / "config" / "performance_players.json", prompts)
        run_config = {
            **environment,
            "implementation": "Performance",
            "contract_id": stable_contract_id,
            "contract_dir": str(contract_dir),
            "customer": args.customer,
            "supplier": args.supplier,
            "seed": args.seed,
            **treatment_metadata(args.goal_treatment, treated_role),
        }
        write_json(output_dir / "config" / "run_config.json", run_config)
        try:
            result = run_performance(
                customer=customer,
                supplier=supplier,
                terms=terms,
                environment_id=args.env,
                seed=args.seed,
            )
            summary = {**run_config, "valid": True, **asdict(result)}
            terminal_status = "complete"
        except EmptyModelResponseError as exc:
            summary = {
                **run_config,
                "valid": False,
                "invalid_reason": "empty_model_response",
                "provider": exc.provider,
                "model": exc.model_name,
                "diagnostics": exc.diagnostics,
            }
            terminal_status = "invalid"
        write_json(output_dir / "data" / "summary.json", summary)
        if runtime is not None:
            save_agent_states(output_dir, runtime.agents, runtime.configs)
        claim.finalize(terminal_status, {"valid": summary["valid"]})
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        with terminate_on_signals():
            return run_worker(args)
    except ProcessTermination as exc:
        return exc.exit_code
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
