#!/usr/bin/env python3
"""Generate a grim-trigger F1 frontier with the modular RC solver.

The base and refined ContractBench-compatible searches live in focused modules;
this file remains the public API and command-line entry point.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pareto.f1_base_search import F1FrontierSearch
from pareto.f1_common import (
    DEFAULT_ENVIRONMENT_ROOT,
    DEFAULT_INITIAL_CONTRACT_PATH,
    DEFAULT_MAX_CUSTOMER_GRID,
    DEFAULT_MAX_SUPPLIER_GRID,
    IMPLEMENTATION_FINGERPRINT,
    SCHEMA_VERSION,
    Contract,
    ContractCandidate,
    Evaluation,
    F1RefinedSearchConfig,
    F1SearchConfig,
    FrontierPoint,
    RefinedContractCandidate,
    SearchCandidate,
    SearchCheckpoint,
    Sweep,
    file_fingerprint,
    load_checkpoint,
    load_environment_selection,
    parse_epsilon_grid,
    resolve_environment_path,
    save_checkpoint_atomic,
    save_json_atomic,
)
from pareto.f1_refined_search import F1FrontierSearchRefined


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "pareto-search"


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


__all__ = (
    "DEFAULT_ENVIRONMENT_ROOT",
    "DEFAULT_INITIAL_CONTRACT_PATH",
    "IMPLEMENTATION_FINGERPRINT",
    "SCHEMA_VERSION",
    "ContractCandidate",
    "Evaluation",
    "F1FrontierSearch",
    "F1FrontierSearchRefined",
    "F1RefinedSearchConfig",
    "F1SearchConfig",
    "FrontierPoint",
    "RefinedContractCandidate",
    "SearchCandidate",
    "SearchCheckpoint",
    "Sweep",
    "file_fingerprint",
    "load_checkpoint",
    "load_environment_selection",
    "main",
    "parse_epsilon_grid",
    "resolve_environment_path",
    "save_checkpoint_atomic",
    "save_json_atomic",
)


def _default_grid(sweep: Sweep, environment_id: int) -> tuple[float, ...]:
    ranges = {
        3: {Sweep.MAX_CUSTOMER: (-40, 90), Sweep.MAX_SUPPLIER: (180, 380)},
        4: {Sweep.MAX_CUSTOMER: (-40, 150), Sweep.MAX_SUPPLIER: (190, 400)},
        5: {Sweep.MAX_CUSTOMER: (-40, 50), Sweep.MAX_SUPPLIER: (180, 240)},
        6: {Sweep.MAX_CUSTOMER: (-40, 120), Sweep.MAX_SUPPLIER: (180, 370)},
    }
    if environment_id in ranges:
        start, stop = ranges[environment_id][sweep]
        return tuple(float(value) for value in range(start, stop + 1, 10))
    return (
        DEFAULT_MAX_CUSTOMER_GRID
        if sweep is Sweep.MAX_CUSTOMER
        else DEFAULT_MAX_SUPPLIER_GRID
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    environment_group = parser.add_mutually_exclusive_group(required=True)
    environment_group.add_argument(
        "--env",
        type=int,
        metavar="ID",
        help="Load envs/ID/environment.json",
    )
    environment_group.add_argument(
        "--environment",
        type=Path,
        help="Load an environment from an explicit JSON path",
    )
    parser.add_argument(
        "--environment-root",
        type=Path,
        help=(
            "Directory containing ID/environment.json folders for --env "
            "(default: RatConBench/envs)"
        ),
    )
    parser.add_argument(
        "--initial-contract",
        type=Path,
        help=(
            "Override algorithms/pareto/examples/f1_initial_contract.json; required with "
            "--refined"
        ),
    )
    parser.add_argument(
        "--sweep", choices=tuple(sweep.value for sweep in Sweep), required=True
    )
    parser.add_argument("--epsilon-grid", type=parse_epsilon_grid)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refined",
        action="store_true",
        help="Use ContractBench's anchor-local targeted-refinement algorithm",
    )
    parser.add_argument("--sat-thresh", type=float, default=0.95)
    parser.add_argument("--p-sat-floor", type=float, default=0.95)
    parser.add_argument("--starts", type=int)
    parser.add_argument("--coarse-iterations", type=int)
    parser.add_argument("--fine-iterations", type=int)
    parser.add_argument("--prune-margin", type=float)
    parser.add_argument(
        "--random-start-attempts",
        type=int,
        help="Upward override for the valid-candidate evaluation cap",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--payment-floor", type=int, default=100)
    parser.add_argument("--payment-cap", type=int)
    parser.add_argument("--search-mode", choices=("best", "first"), default="best")
    parser.add_argument("--quantity-radius", type=int, default=1)
    parser.add_argument("--price-radius", type=float, default=1.0)
    parser.add_argument("--payment-radius", type=int, default=5)
    parser.add_argument("--refined-iterations", type=int, default=12)
    args = parser.parse_args()

    if args.refined and args.epsilon_grid is None:
        parser.error("--refined requires the anchor-specific --epsilon-grid")
    if args.refined and args.initial_contract is None:
        parser.error("--refined requires --initial-contract")
    base_only_options = (
        ("--starts", args.starts),
        ("--coarse-iterations", args.coarse_iterations),
        ("--fine-iterations", args.fine_iterations),
        ("--prune-margin", args.prune_margin),
        ("--random-start-attempts", args.random_start_attempts),
        ("--seed", args.seed),
    )
    if args.refined:
        supplied_base_options = [
            option for option, value in base_only_options if value is not None
        ]
        if supplied_base_options:
            parser.error(
                "--refined does not support base-search options: "
                + ", ".join(supplied_base_options)
            )
    if args.environment_root is not None and args.env is None:
        parser.error("--environment-root requires --env")
    try:
        environment_path, environment = load_environment_selection(
            environment_id=args.env,
            environment_path=args.environment,
            environment_root=(
                DEFAULT_ENVIRONMENT_ROOT
                if args.environment_root is None
                else args.environment_root
            ),
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    initial_contract_path = args.initial_contract or DEFAULT_INITIAL_CONTRACT_PATH
    contract = Contract.from_json(initial_contract_path.resolve())
    payment_cap = (
        int(environment.customer.starting_budget)
        if args.payment_cap is None
        else args.payment_cap
    )
    sweep = Sweep(args.sweep)
    output_dir = DEFAULT_RESULTS_ROOT / f"env-{environment.environment_id:02d}"
    checkpoint_path = (
        args.checkpoint or output_dir / f"{sweep.value}-f1.checkpoint.json"
    ).resolve()
    output_path = (args.output or output_dir / f"{sweep.value}-f1.json").resolve()
    log_path = (args.log or output_dir / f"{sweep.value}-f1.log").resolve()
    common_config: dict[str, Any] = {
        "sweep": sweep,
        "epsilon_grid": args.epsilon_grid
        or _default_grid(sweep, environment.environment_id),
        "satisfaction_threshold": args.sat_thresh,
        "minimum_satisfaction": args.p_sat_floor,
        "payment_floor": args.payment_floor,
        "payment_cap": payment_cap,
        "search_mode": args.search_mode,
    }
    if args.refined:
        config: F1SearchConfig = F1RefinedSearchConfig(
            **common_config,
            number_of_starts=1,
            quantity_radius=args.quantity_radius,
            price_radius=args.price_radius,
            payment_radius=args.payment_radius,
            refined_iterations=args.refined_iterations,
        )
        initial_candidate: SearchCandidate = RefinedContractCandidate.from_contract(
            contract, environment.supplier.products
        )
        search_type = F1FrontierSearchRefined
    else:
        base_overrides = {
            name: value
            for name, value in (
                ("number_of_starts", args.starts),
                ("coarse_iterations", args.coarse_iterations),
                ("fine_iterations", args.fine_iterations),
                ("prune_margin", args.prune_margin),
                ("random_start_attempts", args.random_start_attempts),
                ("seed", args.seed),
            )
            if value is not None
        }
        config = F1SearchConfig(
            **common_config,
            **base_overrides,
        )
        initial_candidate = ContractCandidate.from_contract(
            contract, environment.supplier.products
        )
        search_type = F1FrontierSearch

    environment_digest = file_fingerprint(environment_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log_file:
        try:
            with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(
                _Tee(sys.stderr, log_file)
            ):
                print(
                    f"Starting env {environment.environment_id} {sweep.value}; "
                    f"checkpoint={checkpoint_path}; output={output_path}",
                    flush=True,
                )
                from rc_solver import RCCSolver

                solver = RCCSolver(environment)
                search = search_type(
                    solver,
                    environment,
                    config,
                    initial_candidate,
                    checkpoint_path,
                    environment_fingerprint=environment_digest,
                )
                points = search.run(resume=args.resume)
                save_json_atomic(output_path, search.result_payload(points))
                print(
                    f"Saved {len(points)} {config.sweep.value} F1 candidates to "
                    f"{output_path}",
                    flush=True,
                )
        except BaseException:
            traceback.print_exc(file=log_file)
            raise


if __name__ == "__main__":
    main()
