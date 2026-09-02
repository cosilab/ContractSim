#!/usr/bin/env python3
"""Deduplicate parsed proposals, solve them, and save contract-quality artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
INFERENCES_ROOT = HERE.parent / "inferences"
DEFAULT_ENV_ROOT = REPO_ROOT / "envs"
SOLVER_ROOT = REPO_ROOT / "algorithms" / "rc-solver"
LEGACY_SOLVER_ROOT = (
    REPO_ROOT / "algorithms" / "solver-comparison-colab-full" / "bound"
)
DEFAULT_LEGACY_KERNEL_ROOT = (
    REPO_ROOT / "algorithms" / "solver-comparison-colab-full" / "kernels"
)
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from bellman import ProductionRoundInput  # noqa: E402
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
from contract_completeness import METRIC_VERSION, occupancy_completeness  # noqa: E402
from environment import Environment  # noqa: E402
from rc_solver import RCCSolver  # noqa: E402
from runtime import TransitionRuntime  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.95
MAX_DEFICIT = 2
SUPPORTED = {"substitution", "payment_deduction", "rollover", "grim_trigger"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    path.write_text("".join(line + "\n" for line in lines))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ContingencySpecification:
    substitution: bool
    payment_deduction: bool
    rollover: bool
    minimum_quantities: Mapping[str, int]
    deduction_prices: Mapping[str, float]
    max_deficit: int

    @property
    def combination(self) -> str:
        flags = (
            (self.substitution, "substitution"),
            (self.payment_deduction, "payment_deduction"),
            (self.rollover, "rollover"),
        )
        return "_".join([name for enabled, name in flags if enabled] + ["grim_trigger"])

    def as_mapping(self) -> dict[str, Any]:
        active = [name for enabled, name in (
            (self.substitution, "substitution"),
            (self.payment_deduction, "payment_deduction"),
            (self.rollover, "rollover"),
            (True, "grim_trigger"),
        ) if enabled]
        params: dict[str, Any] = {}
        if self.substitution:
            params["substitution"] = {"min_qty": dict(self.minimum_quantities)}
        if self.payment_deduction:
            params["payment_deduction"] = {"deduction_prices": dict(self.deduction_prices)}
        if self.rollover:
            params["rollover"] = {"max_deficit": self.max_deficit}
        return {"combination": self.combination, "contingencies": active, "params": params}

    @classmethod
    def from_values(
        cls,
        active: Sequence[Any],
        params: Mapping[str, Any],
        contract: Contract,
        environment: Environment,
    ) -> ContingencySpecification:
        names = {str(value).strip().lower().replace("-", "_") for value in active}
        if names - SUPPORTED:
            raise ValueError(f"unsupported contingencies: {sorted(names - SUPPORTED)}")
        sub = params.get("substitution")
        sub = sub if isinstance(sub, Mapping) else {}
        minimums = sub.get("min_qty")
        if not isinstance(minimums, Mapping):
            minimums = environment.customer.minimum_products_per_production_round
        roll = params.get("rollover")
        roll = roll if isinstance(roll, Mapping) else {}
        deficit = roll.get("max_deficit", MAX_DEFICIT)
        deficit = MAX_DEFICIT if deficit is None else int(deficit)
        return cls(
            "substitution" in names,
            "payment_deduction" in names,
            "rollover" in names,
            {product: int(minimums[product]) for product in environment.supplier.products},
            dict(contract.product_prices),
            min(max(deficit, 0), MAX_DEFICIT),
        )


@dataclass
class SolveJob:
    solve_hash: str
    environment_id: int
    contract_source: dict[str, Any]
    specification: ContingencySpecification
    satisfaction_threshold: float
    record_ids: list[str]
    accepted_final_record_ids: list[str]

    @property
    def needs_policy(self) -> bool:
        return bool(self.accepted_final_record_ids)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "solve_hash": self.solve_hash,
            "environment_id": self.environment_id,
            "contract": self.contract_source,
            "contingency": self.specification.as_mapping(),
            "satisfaction_threshold": self.satisfaction_threshold,
            "record_ids": self.record_ids,
            "accepted_final_record_ids": self.accepted_final_record_ids,
            "needs_policy": self.needs_policy,
        }


def contract_source(terms: Mapping[str, Any]) -> dict[str, Any]:
    prices = terms["dish_prices"]
    products = tuple(prices)
    contract = Contract({
        "product_prices": dict(prices),
        "production_schedule": [{
            "week": row["week"],
            "product_quantities": {product: row[product] for product in products},
        } for row in terms["production_schedule"]],
        "payment_schedule": [
            {"week": row["week"], "amount": row["amount"]}
            for row in terms["payment_schedule"]
        ],
    })
    return {
        "product_prices": dict(contract.product_prices),
        "production_schedule": [{
            "week": row.week,
            "product_quantities": dict(row.product_quantities),
        } for row in contract.production_schedule],
        "payment_schedule": [
            {"week": row.week, "amount": row.amount} for row in contract.payment_schedule
        ],
    }


def discover_records(
    parsed_root: Path,
    env_root: Path,
    environment_filter: set[int] | None,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, Environment]]:
    indexes = sorted(parsed_root.glob("**/data/contract_terms/index.json"))
    if not indexes:
        raise FileNotFoundError(f"no parsed proposal indexes under {parsed_root}")
    records: list[dict[str, Any]] = []
    environments: dict[int, Environment] = {}
    for index_path in indexes:
        parts = index_path.relative_to(parsed_root).parts
        environment_id = int(parts[0].removeprefix("env-"))
        if environment_filter and environment_id not in environment_filter:
            continue
        if environment_id not in environments:
            path = env_root / str(environment_id) / "environment.json"
            environments[environment_id] = Environment.from_json(path)
        environment = environments[environment_id]
        for item in read_json(index_path).get("records", []):
            terms_path = index_path.parent / item["terms_file"]
            base = {
                "record_id": str(item["record_id"]),
                "environment_id": environment_id,
                "run": parts[1],
                "matchup": parts[2],
                "offer_index": item.get("offer_index"),
                "round": item.get("round"),
                "is_accepted_final": bool(item.get("is_accepted_final")),
            }
            try:
                terms = read_json(terms_path)["structured_terms"]
                source = contract_source(terms)
                raw_params = terms.get("contingency_params")
                spec = ContingencySpecification.from_values(
                    terms.get("contingency_set") or terms.get("contingencies") or [],
                    raw_params if isinstance(raw_params, Mapping) else {},
                    Contract(source),
                    environment,
                )
                identity = {
                    "schema_version": SCHEMA_VERSION,
                    "environment_id": environment_id,
                    "contract": source,
                    "contingency": spec.as_mapping(),
                    "satisfaction_threshold": threshold,
                }
                solve_hash = hashlib.sha256(canonical(identity).encode()).hexdigest()
                records.append({
                    **base,
                    "solve_hash": solve_hash,
                    "contract": source,
                    "contingency": spec.as_mapping(),
                    "satisfaction_threshold": threshold,
                    "contract_error": None,
                })
            except Exception as error:
                records.append({
                    **base,
                    "solve_hash": None,
                    "contract_error": f"{type(error).__name__}: {error}",
                })
    return records, environments


def deduplicate_jobs(
    records: Sequence[Mapping[str, Any]],
    environments: Mapping[int, Environment],
) -> list[SolveJob]:
    jobs: dict[str, SolveJob] = {}
    for record in records:
        solve_hash = record.get("solve_hash")
        if not solve_hash:
            continue
        environment_id = int(record["environment_id"])
        source = dict(record["contract"])
        raw = record["contingency"]
        spec = ContingencySpecification.from_values(
            raw["contingencies"], raw["params"], Contract(source), environments[environment_id]
        )
        job = jobs.setdefault(str(solve_hash), SolveJob(
            str(solve_hash), environment_id, source, spec,
            float(record["satisfaction_threshold"]), [], [],
        ))
        record_id = str(record["record_id"])
        job.record_ids.append(record_id)
        if record.get("is_accepted_final"):
            job.accepted_final_record_ids.append(record_id)
    return sorted(jobs.values(), key=lambda job: (job.environment_id, job.solve_hash))


def make_contingency_solver(environment: Environment, spec: ContingencySpecification) -> Any:
    key = (spec.substitution, spec.payment_deduction, spec.rollover)
    if key == (False, False, False):
        return RCCSolver(environment)
    classes = {
        (True, False, False): SubstitutionOneStepSolver,
        (False, True, False): PaymentDeductionOneStepSolver,
        (True, True, False): SubstitutionPaymentDeductionOneStepSolver,
        (False, False, True): RolloverOneStepSolver,
        (True, False, True): SubstitutionRolloverOneStepSolver,
        (False, True, True): PaymentDeductionRolloverOneStepSolver,
        (True, True, True): SubstitutionPaymentDeductionRolloverOneStepSolver,
    }
    args: list[Any] = [environment]
    kwargs: dict[str, Any] = {}
    if spec.payment_deduction:
        args.append(spec.deduction_prices)
    if spec.substitution:
        kwargs["minimum_quantities"] = spec.minimum_quantities
    if spec.rollover:
        kwargs["max_deficit"] = spec.max_deficit
    return classes[key](*args, **kwargs)


def round_inputs(contract: Contract, runtime: TransitionRuntime) -> tuple[Any, ...]:
    transitions = runtime.for_contract(contract)
    return tuple(ProductionRoundInput(
        production=production,
        payment=payment,
        transitions=transitions.for_week(production.week),
    ) for payment, production in zip(
        contract.payment_schedule[:-1], contract.production_schedule
    ))


def initial_index(environment: Environment, spec: ContingencySpecification) -> tuple[int, ...]:
    prefix = (
        (0,) * len(environment.supplier.products)
        if spec.rollover else ((0,) if spec.payment_deduction else ())
    )
    return prefix + (0,) * len(environment.supplier.resources) + (
        environment.supplier.starting_capital,
    )


def scalar_result(
    state: Any, environment: Environment, spec: ContingencySpecification
) -> dict[str, float]:
    state.stacked().block_until_ready()
    index = initial_index(environment, spec)
    return {
        "P_sat": float(state.satisfaction_probability[index]),
        "E_customer_utility": float(
            environment.customer.starting_budget
            + state.customer_value[index]
            - state.total_paid[index]
        ),
        "E_caterer_utility": float(state.supplier_utility[index]),
    }


def legacy_contract(job: SolveJob) -> dict[str, Any]:
    contract = Contract(job.contract_source)
    contingency = job.specification.as_mapping()
    return {
        "contract_qty_schedule": [
            dict(round_.product_quantities)
            for round_ in contract.production_schedule
        ],
        "dish_prices": dict(contract.product_prices),
        "payments": [round_.amount for round_ in contract.payment_schedule[:-1]],
        "post_game_payment": contract.payment_schedule[-1].amount,
        "sat_thresh": job.satisfaction_threshold,
        "contingencies": contingency["contingencies"],
        "contingency_params": contingency["params"],
    }


def solve_legacy_job(job: SolveJob, solver: Any) -> dict[str, Any]:
    result = solver.solve_ev(legacy_contract(job))
    return {
        "solve_hash": job.solve_hash,
        "environment_id": job.environment_id,
        "solver_status": "solved",
        "solver_backend": "legacy_bound",
        "P_sat": float(result[0]),
        "E_caterer_utility": float(result[1]),
        "E_customer_utility": float(result[6]),
        "policy_path": None,
        "completeness_path": None,
    }


def policy_axes(environment: Environment, spec: ContingencySpecification) -> list[str]:
    axes = ["round"]
    if spec.rollover:
        axes += [f"deficit_{name.lower()}" for name in environment.supplier.products]
    elif spec.payment_deduction:
        axes.append("previous_shortfall")
    axes += [f"inventory_{name.lower()}" for name in environment.supplier.resources]
    return axes + ["capital", "price_scenario", "resource"]


def solve_job(
    job: SolveJob,
    environment: Environment,
    runtime: TransitionRuntime,
    solver: Any,
    output: Path,
    compute_completeness: bool,
    existing_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = Contract(job.contract_source)
    policy_dir = output / "policies" / f"env-{job.environment_id:02d}" / job.solve_hash
    policy_path = policy_dir / "policy.npz"
    completeness_path = policy_dir / "completeness.json"
    needs_policy = job.needs_policy
    reuse_policy = bool(
        needs_policy
        and compute_completeness
        and existing_result
        and all(
            key in existing_result
            for key in ("P_sat", "E_customer_utility", "E_caterer_utility")
        )
        and policy_path.is_file()
    )
    if reuse_policy:
        scalar = {
            key: float(existing_result[key])
            for key in ("P_sat", "E_customer_utility", "E_caterer_utility")
        }
        with np.load(policy_path) as artifact:
            policies = artifact["policies"]
        backend = solver.bellman_solver if isinstance(solver, RCCSolver) else solver
        rounds = round_inputs(
            contract,
            solver.transition_runtime if isinstance(solver, RCCSolver) else runtime,
        )
    elif isinstance(solver, RCCSolver):
        scalar = solver.solve(
            contract,
            job.satisfaction_threshold,
            policy_path=policy_path if needs_policy else None,
        )
        scalar["E_caterer_utility"] = scalar.pop("E_supplier_utility")
        if needs_policy:
            with np.load(policy_path) as artifact:
                policies = artifact["policies"]
        else:
            policies = None
        backend = solver.bellman_solver
        rounds = round_inputs(contract, solver.transition_runtime)
    else:
        backend = solver
        rounds = round_inputs(contract, runtime)
        if needs_policy:
            state, policies = solver.solve_rounds_from_final_payment_with_policy(
                rounds, contract.payment_schedule[-1], job.satisfaction_threshold
            )
        else:
            state = solver.solve_rounds_from_final_payment(
                rounds, contract.payment_schedule[-1], job.satisfaction_threshold
            )
            policies = None
        scalar = scalar_result(state, environment, job.specification)
    result = {
        "solve_hash": job.solve_hash,
        "environment_id": job.environment_id,
        "solver_status": "solved",
        **scalar,
        "policy_path": (
            str(policy_path.relative_to(output))
            if policies is not None or policy_path.is_file() else None
        ),
        "completeness_path": (
            str(completeness_path.relative_to(output))
            if (policies is not None and compute_completeness)
            or completeness_path.is_file() else None
        ),
    }
    if policies is None:
        return result
    if not reuse_policy:
        policy_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            policy_path,
            policies=policies,
            policy_axes=np.array(json.dumps(policy_axes(environment, job.specification))),
            solve_hash=np.array(job.solve_hash),
            price_scenarios=np.asarray(backend.price_kernel.scenarios),
            price_probabilities=np.asarray(backend.price_kernel.probabilities),
        )
    if not compute_completeness:
        return result
    metric = occupancy_completeness(
        policies,
        solver=backend,
        round_inputs=rounds,
        final_payment=contract.payment_schedule[-1],
        specification=job.specification,
        starting_capital=environment.supplier.starting_capital,
        p_sat=scalar["P_sat"],
    )
    write_json(completeness_path, {
        "schema_version": SCHEMA_VERSION,
        "solve_hash": job.solve_hash,
        "policy_path": str(policy_path.relative_to(output)),
        **metric,
    })
    return result


def run_jobs(
    jobs: Sequence[SolveJob],
    environments: Mapping[int, Environment],
    output: Path,
    force: bool,
    compute_completeness: bool,
    legacy_solver: bool = False,
    env_root: Path | None = None,
    legacy_kernel_root: Path | None = None,
) -> list[dict[str, Any]]:
    if legacy_solver and compute_completeness:
        raise ValueError("The legacy solver run requires --skip-completeness")
    if legacy_solver:
        if str(LEGACY_SOLVER_ROOT) not in sys.path:
            sys.path.insert(0, str(LEGACY_SOLVER_ROOT))
        from solver_rcc_bound import BoundRCCSolver

    grouped: dict[int, list[SolveJob]] = defaultdict(list)
    for job in jobs:
        grouped[job.environment_id].append(job)

    def run_environments(environment_ids: Sequence[int], device: Any = None) -> list[dict[str, Any]]:
        import jax

        lane_results = []
        with jax.default_device(device):
            for environment_id in environment_ids:
                environment_jobs = grouped[environment_id]
                environment = environments[environment_id]
                runtime = TransitionRuntime(environment)
                solvers: dict[str, Any] = {}
                legacy = None
                if legacy_solver:
                    assert env_root is not None and legacy_kernel_root is not None
                    legacy = BoundRCCSolver(
                        env_root / str(environment_id) / "environment.json",
                        legacy_kernel_root / f"env-{environment_id:02d}.npz",
                    )
                for number, job in enumerate(environment_jobs, 1):
                    path = output / "solves" / f"env-{environment_id:02d}" / f"{job.solve_hash}.json"
                    existing_result = None
                    if path.is_file() and not force:
                        result = read_json(path)
                        existing_result = result
                        scalars = ("P_sat", "E_customer_utility", "E_caterer_utility")
                        policy_ready = (
                            not job.needs_policy
                            or bool(result.get("policy_path") and (output / result["policy_path"]).is_file())
                        )
                        completeness_ready = (
                            not job.needs_policy
                            or not compute_completeness
                            or bool(
                                result.get("completeness_path")
                                and (output / result["completeness_path"]).is_file()
                            )
                        )
                        if result.get("solver_status") == "solved" and all(
                            key in result for key in scalars
                        ) and policy_ready and completeness_ready:
                            lane_results.append(result)
                            continue
                    print(f"[env {environment_id} {number}/{len(environment_jobs)}] {job.solve_hash[:12]}", flush=True)
                    try:
                        if legacy is not None:
                            result = solve_legacy_job(job, legacy)
                        else:
                            key = canonical(job.specification.as_mapping())
                            if key not in solvers:
                                solvers[key] = make_contingency_solver(environment, job.specification)
                            result = solve_job(
                                job, environment, runtime, solvers[key], output,
                                compute_completeness, existing_result,
                            )
                    except Exception as error:
                        result = {
                            "solve_hash": job.solve_hash,
                            "environment_id": job.environment_id,
                            "solver_status": f"failed:{type(error).__name__}",
                            "solver_error": str(error),
                            "policy_path": None,
                            "completeness_path": None,
                        }
                    write_json(path, result)
                    lane_results.append(result)
        return lane_results

    environment_ids = sorted(grouped)
    import jax

    gpu_devices = jax.devices("gpu") if jax.default_backend() == "gpu" else []
    if not legacy_solver and len(gpu_devices) >= 2:
        odd_environments = [environment_id for environment_id in environment_ids if environment_id % 2]
        even_environments = [environment_id for environment_id in environment_ids if not environment_id % 2]
        print("Odd environments -> GPU 0; even environments -> GPU 1", flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            odd = executor.submit(run_environments, odd_environments, gpu_devices[0])
            even = executor.submit(run_environments, even_environments, gpu_devices[1])
            results = odd.result() + even.result()
    else:
        results = run_environments(environment_ids)
    return sorted(results, key=lambda result: (result["environment_id"], result["solve_hash"]))


def index_records(
    records: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_hash = {result["solve_hash"]: result for result in results}
    indexed = []
    for source in records:
        record = dict(source)
        for key in ("contract", "contingency", "satisfaction_threshold"):
            record.pop(key, None)
        result = by_hash.get(record.get("solve_hash"))
        record["solver_status"] = (
            result["solver_status"] if result
            else "contract_invalid" if not record.get("solve_hash") else "not_selected"
        )
        record["result_path"] = (
            f"solves/env-{record['environment_id']:02d}/{record['solve_hash']}.json"
            if result else None
        )
        record["policy_path"] = result.get("policy_path") if result else None
        record["completeness_path"] = result.get("completeness_path") if result else None
        indexed.append(record)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--parsed-root", type=Path)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env", type=int, action="append", choices=range(1, 7))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--skip-completeness", action="store_true")
    parser.add_argument("--legacy-solver", action="store_true")
    parser.add_argument(
        "--legacy-kernel-root",
        type=Path,
        default=DEFAULT_LEGACY_KERNEL_ROOT,
    )
    parser.add_argument("--sat-thresh", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.sat_thresh <= 1:
        parser.error("--sat-thresh must be between zero and one")

    campaign_root = INFERENCES_ROOT / args.campaign
    parsed_root = (args.parsed_root or campaign_root / "parsed").resolve()
    output = (args.output or campaign_root / "contract-qual").resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, environments = discover_records(
        parsed_root, args.env_root.resolve(), set(args.env) if args.env else None,
        args.sat_thresh,
    )
    jobs = deduplicate_jobs(records, environments)
    jobs = jobs[: args.limit] if args.limit else jobs
    write_jsonl(output / "jobs.jsonl", [job.as_mapping() for job in jobs])
    results: list[dict[str, Any]] = []
    if not args.prepare_only:
        import jax

        if args.require_gpu and jax.default_backend() != "gpu":
            raise RuntimeError(f"GPU required; JAX backend is {jax.default_backend()}")
        results = run_jobs(
            jobs, environments, output, args.force,
            not args.skip_completeness,
            args.legacy_solver,
            args.env_root.resolve(),
            args.legacy_kernel_root.resolve(),
        )
    write_jsonl(output / "records.jsonl", index_records(records, results))
    write_json(output / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "campaign": args.campaign,
        "n_records": len(records),
        "n_invalid": sum(not record.get("solve_hash") for record in records),
        "n_unique_jobs": len(jobs),
        "n_policy_jobs": sum(job.needs_policy for job in jobs),
        "n_solved": sum(result.get("solver_status") == "solved" for result in results),
        "n_failed": sum(result.get("solver_status") != "solved" for result in results),
        "prepare_only": args.prepare_only,
        "compute_completeness": not args.skip_completeness,
        "solver_backend": "legacy_bound" if args.legacy_solver else "modular",
        "completeness_metric_version": METRIC_VERSION,
    })
    print(f"{'Prepared' if args.prepare_only else 'Solved'} {len(jobs)} jobs under {output}")


if __name__ == "__main__":
    main()
