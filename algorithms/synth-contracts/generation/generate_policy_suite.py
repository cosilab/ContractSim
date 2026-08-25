#!/usr/bin/env python3
"""Generate the 240 contingency-aware RCC policy artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS_ROOT = REPOSITORY_ROOT / "algorithms"
RC_SOLVER_ROOT = ALGORITHMS_ROOT / "rc-solver"
if str(RC_SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(RC_SOLVER_ROOT))

from bellman import OneStepSolver, ProductionRoundInput  # noqa: E402
from contingencies import (  # noqa: E402
    PaymentDeductionOneStepSolver,
    PaymentDeductionRolloverOneStepSolver,
    RolloverOneStepSolver,
    SubstitutionOneStepSolver,
    SubstitutionPaymentDeductionOneStepSolver,
    SubstitutionPaymentDeductionRolloverOneStepSolver,
    SubstitutionRolloverOneStepSolver,
)
from contract import Contract  # noqa: E402
from environment import Environment  # noqa: E402
from runtime import TransitionRuntime  # noqa: E402


SATISFACTION_THRESHOLD = 0.95
SOURCE_BANDS = ("p050", "p060", "p070", "p080", "p090")
EVALUATION_ENVIRONMENTS = {
    3: (1, 3),
    4: (2, 4),
    5: (5,),
    6: (6,),
}


@dataclass(frozen=True)
class ContingencySpec:
    name: str
    substitution: bool = False
    payment_deduction: bool = False
    rollover: bool = False

    @property
    def mode(self) -> str:
        if self.payment_deduction and self.rollover:
            return "payment_deduction_rollover"
        if self.payment_deduction:
            return "payment_deduction"
        if self.rollover:
            return "rollover"
        return "grim_trigger"

    @property
    def contingencies(self) -> list[str]:
        names = []
        if self.substitution:
            names.append("substitution")
        if self.payment_deduction:
            names.append("payment_deduction")
        if self.rollover:
            names.append("rollover")
        names.append("grim_trigger")
        return names


CONTINGENCY_SPECS = (
    ContingencySpec("grim_trigger"),
    ContingencySpec("substitution", substitution=True),
    ContingencySpec("payment_deduction", payment_deduction=True),
    ContingencySpec(
        "substitution_payment_deduction",
        substitution=True,
        payment_deduction=True,
    ),
    ContingencySpec("rollover", rollover=True),
    ContingencySpec(
        "substitution_rollover",
        substitution=True,
        rollover=True,
    ),
    ContingencySpec(
        "payment_deduction_rollover",
        payment_deduction=True,
        rollover=True,
    ),
    ContingencySpec(
        "substitution_payment_deduction_rollover",
        substitution=True,
        payment_deduction=True,
        rollover=True,
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contract_from_terms(terms: dict[str, Any]) -> Contract:
    prices = dict(terms["dish_prices"])
    return Contract(
        {
            "product_prices": prices,
            "production_schedule": [
                {
                    "week": int(row["week"]),
                    "product_quantities": {
                        product: int(row[product]) for product in prices
                    },
                }
                for row in terms["production_schedule"]
            ],
            "payment_schedule": list(terms["payment_schedule"]),
        }
    )


def _round_inputs(
    contract: Contract,
    runtime: TransitionRuntime,
) -> tuple[ProductionRoundInput, ...]:
    transitions = runtime.for_contract(contract)
    return tuple(
        ProductionRoundInput(
            production=production,
            payment=payment,
            transitions=transitions.for_week(production.week),
        )
        for payment, production in zip(
            contract.payment_schedule[:-1],
            contract.production_schedule,
        )
    )


def _solver_for(
    spec: ContingencySpec,
    environment: Environment,
    contract: Contract,
) -> OneStepSolver:
    prices = contract.product_prices
    if spec.payment_deduction and spec.rollover:
        solver_type = (
            SubstitutionPaymentDeductionRolloverOneStepSolver
            if spec.substitution
            else PaymentDeductionRolloverOneStepSolver
        )
        return solver_type(environment, prices, max_deficit=2)
    if spec.payment_deduction:
        solver_type = (
            SubstitutionPaymentDeductionOneStepSolver
            if spec.substitution
            else PaymentDeductionOneStepSolver
        )
        return solver_type(environment, prices)
    if spec.rollover:
        solver_type = (
            SubstitutionRolloverOneStepSolver
            if spec.substitution
            else RolloverOneStepSolver
        )
        return solver_type(environment, max_deficit=2)
    if spec.substitution:
        return SubstitutionOneStepSolver(environment)
    return OneStepSolver(environment)


def _policy_axes(
    spec: ContingencySpec,
    environment: Environment,
) -> list[str]:
    axes = ["round"]
    if spec.payment_deduction and not spec.rollover:
        axes.append("had_prior_shortfall")
    if spec.rollover:
        axes.extend(
            f"deficit_{product}" for product in environment.supplier.products
        )
    axes.extend(
        f"inv_{resource.lower()}" for resource in environment.supplier.resources
    )
    axes.extend(("budget", "scenario", "ingredient"))
    return axes


def _contingency_params(
    spec: ContingencySpec,
    environment: Environment,
    contract: Contract,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if spec.substitution:
        params["substitution"] = {
            "min_qty": dict(
                environment.customer.minimum_products_per_production_round
            )
        }
    if spec.payment_deduction:
        params["payment_deduction"] = {
            "deduction_prices": dict(contract.product_prices)
        }
    if spec.rollover:
        params["rollover"] = {"max_deficit": 2}
    return params


def _metrics(state: Any, environment: Environment) -> dict[str, float]:
    state.stacked().block_until_ready()
    initial_state = (
        (0,) * (len(state.shape) - 1)
        + (environment.supplier.starting_capital,)
    )
    customer_value = float(state.customer_value[initial_state])
    total_paid = float(state.total_paid[initial_state])
    return {
        "P_sat": float(state.satisfaction_probability[initial_state]),
        "E_supplier_utility": float(state.supplier_utility[initial_state]),
        "E_customer_utility": (
            float(environment.customer.starting_budget)
            + customer_value
            - total_paid
        ),
    }


def _save_policy(
    path: Path,
    policy: np.ndarray,
    axes: list[str],
    solver: OneStepSolver,
    satisfaction_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            policies=policy,
            policy_axes=np.array(json.dumps(axes)),
            PRICE_RANGES=np.asarray(
                solver.price_kernel.scenarios,
                dtype=np.float32,
            ),
            PRICE_PROB=np.asarray(
                solver.price_kernel.probabilities,
                dtype=np.float64,
            ),
            SAT_THRESH=np.float32(satisfaction_threshold),
        )


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    environments = {
        environment_id: Environment.from_json(
            args.environment_root / str(environment_id) / "environment.json"
        )
        for environment_id in range(1, 7)
    }
    runtimes = {
        environment_id: TransitionRuntime(environment)
        for environment_id, environment in environments.items()
    }
    manifest: list[dict[str, Any]] = []
    instance_number = 0

    for source_environment_id, evaluation_ids in EVALUATION_ENVIRONMENTS.items():
        for band in SOURCE_BANDS:
            source_dir = (
                args.source_root
                / f"env-{source_environment_id:02d}"
                / band
            )
            terms = _read_json(source_dir / "terms.json")
            source_metrics = _read_json(source_dir / "result.json")
            contract = _contract_from_terms(terms)

            for evaluation_environment_id in evaluation_ids:
                environment = environments[evaluation_environment_id]
                round_inputs = _round_inputs(
                    contract,
                    runtimes[evaluation_environment_id],
                )
                for spec in CONTINGENCY_SPECS:
                    instance_number += 1
                    instance_id = (
                        f"env-{evaluation_environment_id:02d}_"
                        f"from-env-{source_environment_id:02d}_"
                        f"{band}_{spec.name}"
                    )
                    instance_dir = args.output_root / instance_id
                    policy_path = instance_dir / "baselines" / "rcc" / "policy.npz"
                    result_path = instance_dir / "result.json"
                    if args.resume and policy_path.is_file() and result_path.is_file():
                        print(
                            f"[{instance_number:03d}/240] resume: {instance_id}",
                            flush=True,
                        )
                        manifest.append(_read_json(result_path))
                        continue

                    print(
                        f"[{instance_number:03d}/240] solving: {instance_id}",
                        flush=True,
                    )
                    solver = _solver_for(spec, environment, contract)
                    state, policy = solver.solve_rounds_from_final_payment_with_policy(
                        round_inputs,
                        contract.payment_schedule[-1],
                        args.satisfaction_threshold,
                    )
                    evaluation_metrics = _metrics(state, environment)
                    params = _contingency_params(spec, environment, contract)
                    config = {
                        "schema_version": 1,
                        "source_environment_id": source_environment_id,
                        "evaluation_environment_id": evaluation_environment_id,
                        "environment_id": evaluation_environment_id,
                        "source_band": band,
                        "contingency_variant": spec.name,
                        "contingencies": spec.contingencies,
                        "contingency_mode": spec.mode,
                        "contingency_params": params,
                    }
                    result = {
                        "instance_id": instance_id,
                        **config,
                        "source_generation_metrics": source_metrics,
                        "evaluation_metrics": evaluation_metrics,
                    }
                    _write_json(instance_dir / "terms.json", terms)
                    _write_json(instance_dir / "config.json", config)
                    _save_policy(
                        policy_path,
                        policy,
                        _policy_axes(spec, environment),
                        solver,
                        args.satisfaction_threshold,
                    )
                    _write_json(result_path, result)
                    manifest.append(result)
                    _write_json(args.output_root / "manifest.json", manifest)
                    print(
                        f"[{instance_number:03d}/240] saved:   {instance_id}",
                        flush=True,
                    )

    if len(manifest) != 240:
        raise RuntimeError(f"Expected 240 suite entries, generated {len(manifest)}")
    _write_json(args.output_root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ALGORITHMS_ROOT / "synth-contracts" / "data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ALGORITHMS_ROOT / "synth-contracts" / "data" / "policy-suite",
    )
    parser.add_argument(
        "--environment-root",
        type=Path,
        default=REPOSITORY_ROOT / "envs",
    )
    parser.add_argument(
        "--satisfaction-threshold",
        type=float,
        default=SATISFACTION_THRESHOLD,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    generate(build_parser().parse_args())


if __name__ == "__main__":
    main()
