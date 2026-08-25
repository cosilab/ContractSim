"""Compare RC solver lifecycle strategies on varied weekly requirements."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = CODE_DIR / "tests/data/environment_09.json"
DEFAULT_CONTRACT = CODE_DIR / "tests/data/rollover_contract.json"
SOLVER_PATH = CODE_DIR / "rc_solver.py"
MODES = ("separate-process", "fresh-runtime", "shared-runtime")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _varied_contract(source: dict[str, Any], variant: int) -> dict[str, Any]:
    """Vary every production week's product quantities deterministically."""
    varied = copy.deepcopy(source)
    products = tuple(varied["product_prices"])
    generator = random.Random(10_000 + variant)
    for production_week in varied["production_schedule"]:
        production_week["product_quantities"] = {
            product: generator.randint(1, 3) for product in products
        }
    return varied


def _contract_variants(
    source: dict[str, Any],
    number_of_contracts: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _varied_contract(source, variant)
        for variant in range(number_of_contracts)
    )


def _device_probe(allow_cpu: bool) -> dict[str, Any]:
    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not allow_cpu:
        if backend != "gpu":
            raise RuntimeError(
                "The benchmark requires a GPU; pass --allow-cpu only for a smoke test"
            )
        if len(gpu_devices) != 1:
            raise RuntimeError(
                "Expose exactly one GPU, for example with CUDA_VISIBLE_DEVICES=0"
            )
    return {
        "backend": backend,
        "devices": [str(device) for device in devices],
    }


def _result_from_state(environment: Any, state: Any) -> dict[str, float]:
    state.stacked().block_until_ready()
    initial_state = (0,) * len(environment.supplier.resources) + (
        environment.supplier.starting_capital,
    )
    probability = float(state.satisfaction_probability[initial_state])
    supplier_utility = float(state.supplier_utility[initial_state])
    customer_value = float(state.customer_value[initial_state])
    total_paid = float(state.total_paid[initial_state])
    return {
        "P_sat": probability,
        "E_supplier_utility": supplier_utility,
        "E_customer_utility": (
            float(environment.customer.starting_budget)
            + customer_value
            - total_paid
        ),
    }


def _solve_in_process(
    environment: Any,
    contract_source: dict[str, Any],
    satisfaction_threshold: float,
    *,
    solver: Any | None = None,
    transition_runtime: Any | None = None,
) -> tuple[dict[str, float], dict[str, float], Any, Any]:
    from bellman import OneStepSolver, ProductionRoundInput
    from contract import Contract
    from runtime import TransitionRuntime

    total_start = time.perf_counter()
    setup_start = total_start
    if solver is None:
        solver = OneStepSolver(environment)
    if transition_runtime is None:
        transition_runtime = TransitionRuntime(environment)
    setup_seconds = time.perf_counter() - setup_start

    contract_start = time.perf_counter()
    contract = Contract(contract_source)
    contract_seconds = time.perf_counter() - contract_start

    transition_start = time.perf_counter()
    transitions = transition_runtime.for_contract(contract)
    round_inputs = tuple(
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
    transition_seconds = time.perf_counter() - transition_start

    solve_start = time.perf_counter()
    state = solver.solve_rounds_from_final_payment(
        round_inputs,
        contract.payment_schedule[-1],
        satisfaction_threshold,
    )
    result = _result_from_state(environment, state)
    solve_seconds = time.perf_counter() - solve_start
    return (
        result,
        {
            "total": time.perf_counter() - total_start,
            "setup": setup_seconds,
            "contract": contract_seconds,
            "transitions": transition_seconds,
            "solve": solve_seconds,
        },
        solver,
        transition_runtime,
    )


def _run_in_process_worker(args: argparse.Namespace) -> dict[str, Any]:
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    from environment import Environment

    device = _device_probe(args.allow_cpu)
    environment = Environment.from_json(args.environment)
    base_contract = _load_json(args.base_contract)
    contracts = _contract_variants(base_contract, args.contracts)
    warmup_contract = _varied_contract(base_contract, args.contracts + 17)

    shared_solver = None
    shared_transitions = None
    if args.worker_mode == "shared-runtime":
        from bellman import OneStepSolver
        from runtime import TransitionRuntime

        shared_solver = OneStepSolver(environment)
        shared_transitions = TransitionRuntime(environment)

    warmup_seconds: list[float] = []
    for _ in range(args.warmups):
        start = time.perf_counter()
        _, _, warmed_solver, warmed_transitions = _solve_in_process(
            environment,
            warmup_contract,
            args.sat_thresh,
            solver=shared_solver,
            transition_runtime=shared_transitions,
        )
        warmup_seconds.append(time.perf_counter() - start)
        if args.worker_mode == "shared-runtime":
            shared_solver = warmed_solver
            shared_transitions = warmed_transitions

    contract_timings: list[dict[str, float]] = []
    repeat_seconds: list[float] = []
    reference_results: list[dict[str, float]] | None = None
    for _ in range(args.repeats):
        repeat_start = time.perf_counter()
        current_results: list[dict[str, float]] = []
        for contract_source in contracts:
            result, timing, _, _ = _solve_in_process(
                environment,
                contract_source,
                args.sat_thresh,
                solver=shared_solver,
                transition_runtime=shared_transitions,
            )
            current_results.append(result)
            contract_timings.append(timing)
        repeat_seconds.append(time.perf_counter() - repeat_start)
        if reference_results is None:
            reference_results = current_results

    cache_size = (
        len(shared_transitions.kernels_by_need)
        if shared_transitions is not None
        else None
    )
    return {
        "mode": args.worker_mode,
        "device": device,
        "warmup_seconds": warmup_seconds,
        "repeat_seconds": repeat_seconds,
        "contract_timings": contract_timings,
        "results": reference_results,
        "cache_size": cache_size,
    }


def _run_cli_contract(
    environment_path: Path,
    contract_path: Path,
    satisfaction_threshold: float,
) -> tuple[dict[str, float], float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(SOLVER_PATH),
            "--environment",
            str(environment_path),
            "--contract",
            str(contract_path),
            "--sat-thresh",
            str(satisfaction_threshold),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout), time.perf_counter() - start


def _run_separate_process_worker(args: argparse.Namespace) -> dict[str, Any]:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, jax; "
                "print(json.dumps({'backend': jax.default_backend(), "
                "'devices': [str(d) for d in jax.devices()], "
                "'platforms': [d.platform for d in jax.devices()]}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    device = json.loads(probe.stdout)
    if not args.allow_cpu:
        gpu_devices = [
            platform for platform in device["platforms"] if platform == "gpu"
        ]
        if device["backend"] != "gpu" or len(gpu_devices) != 1:
            raise RuntimeError(
                "Expose exactly one GPU, for example with CUDA_VISIBLE_DEVICES=0"
            )

    base_contract = _load_json(args.base_contract)
    contracts = _contract_variants(base_contract, args.contracts)
    warmup_contract = _varied_contract(base_contract, args.contracts + 17)

    with tempfile.TemporaryDirectory(prefix="rc-runtime-contracts-") as directory:
        contract_directory = Path(directory)
        warmup_path = contract_directory / "warmup.json"
        warmup_path.write_text(json.dumps(warmup_contract))
        contract_paths: list[Path] = []
        for index, contract in enumerate(contracts):
            path = contract_directory / f"contract-{index}.json"
            path.write_text(json.dumps(contract))
            contract_paths.append(path)

        warmup_seconds = [
            _run_cli_contract(
                args.environment,
                warmup_path,
                args.sat_thresh,
            )[1]
            for _ in range(args.warmups)
        ]
        contract_timings: list[dict[str, float]] = []
        repeat_seconds: list[float] = []
        reference_results: list[dict[str, float]] | None = None
        for _ in range(args.repeats):
            repeat_start = time.perf_counter()
            current_results: list[dict[str, float]] = []
            for path in contract_paths:
                result, elapsed = _run_cli_contract(
                    args.environment,
                    path,
                    args.sat_thresh,
                )
                current_results.append(result)
                contract_timings.append({"total": elapsed})
            repeat_seconds.append(time.perf_counter() - repeat_start)
            if reference_results is None:
                reference_results = current_results

    return {
        "mode": args.worker_mode,
        "device": device,
        "warmup_seconds": warmup_seconds,
        "repeat_seconds": repeat_seconds,
        "contract_timings": contract_timings,
        "results": reference_results,
        "cache_size": None,
    }


def _run_worker(args: argparse.Namespace) -> None:
    if args.worker_mode == "separate-process":
        result = _run_separate_process_worker(args)
    else:
        result = _run_in_process_worker(args)
    print(json.dumps(result))


def _assert_result_parity(results: list[dict[str, Any]]) -> None:
    expected = results[0]["results"]
    for mode_result in results[1:]:
        actual = mode_result["results"]
        if len(actual) != len(expected):
            raise RuntimeError("Benchmark modes returned different result counts")
        for contract_index, (left, right) in enumerate(zip(expected, actual)):
            for metric in left:
                if not math.isclose(
                    left[metric],
                    right[metric],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        f"Result mismatch for contract {contract_index}, "
                        f"metric {metric}: {left[metric]} != {right[metric]}"
                    )


def _worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--environment",
        str(args.environment),
        "--base-contract",
        str(args.base_contract),
        "--contracts",
        str(args.contracts),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--sat-thresh",
        str(args.sat_thresh),
    ]
    if args.allow_cpu:
        command.append("--allow-cpu")
    return command


def _run_controller(args: argparse.Namespace) -> None:
    worker_environment = os.environ.copy()
    if args.gpu is not None:
        worker_environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    mode_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rc-jax-cache-") as cache_root:
        for mode in MODES:
            mode_environment = worker_environment.copy()
            mode_cache = Path(cache_root) / mode
            mode_cache.mkdir()
            mode_environment["JAX_COMPILATION_CACHE_DIR"] = str(mode_cache)
            completed = subprocess.run(
                _worker_command(args, mode),
                check=True,
                capture_output=True,
                text=True,
                env=mode_environment,
            )
            mode_results.append(json.loads(completed.stdout))

    _assert_result_parity(mode_results)
    print(
        f"Backend: {mode_results[0]['device']['backend']} | "
        f"contracts: {args.contracts} | repeats: {args.repeats}"
    )
    print("The first warm-up is reported separately and excluded from medians.\n")
    print(
        f"{'mode':<20} {'first warm-up':>14} "
        f"{'median/contract':>16} {'median batch':>14} {'cache':>8}"
    )
    for result in mode_results:
        warmup = result["warmup_seconds"]
        first_warmup = warmup[0] if warmup else float("nan")
        contract_times = [
            timing["total"] for timing in result["contract_timings"]
        ]
        cache = result["cache_size"]
        cache_text = "-" if cache is None else str(cache)
        print(
            f"{result['mode']:<20} {first_warmup:>14.4f} "
            f"{statistics.median(contract_times):>16.4f} "
            f"{statistics.median(result['repeat_seconds']):>14.4f} "
            f"{cache_text:>8}"
        )
    print("\nAll modes returned matching solver results.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark separate processes, fresh runtimes, and a shared runtime"
        )
    )
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contracts", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sat-thresh", type=float, default=0.95)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for workers; alternatively set it externally",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU smoke test instead of requiring exactly one GPU",
    )
    parser.add_argument(
        "--worker-mode",
        choices=MODES,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.contracts < 1 or args.repeats < 1 or args.warmups < 1:
        parser.error("--contracts, --repeats, and --warmups must be positive")
    return args


if __name__ == "__main__":
    parsed_args = _parse_args()
    if parsed_args.worker_mode:
        _run_worker(parsed_args)
    else:
        _run_controller(parsed_args)
