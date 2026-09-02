"""Public entrypoint for single and batch performance runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_utils.llm import PROVIDERS
from performance.contracts import (
    contract_environment_id,
    validate_contract_environment,
)
from performance.orchestration.batch import run_batch
from performance.orchestration.worker import BASELINES
from performance.prompting import GOAL_TREATMENT_CHOICES


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_CONTRACTS_ROOT = PROJECT_ROOT / "algorithms" / "synth-contracts" / "data"
DEFAULT_ENVIRONMENTS = (1, 2, 3, 4, 5, 6)
PARTICIPANTS = (*BASELINES, *PROVIDERS)
EXCLUDED_BATCH_VARIANTS = frozenset(
    {"payment_deduction", "substitution_payment_deduction"}
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--goal-treatment",
        choices=GOAL_TREATMENT_CHOICES,
        default="control",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    single = modes.add_parser("single")
    _shared(single)
    single.add_argument("--env", type=int, required=True, choices=range(1, 7))
    single.add_argument("--contract-dir", type=Path, required=True)
    single.add_argument("--customer", choices=PARTICIPANTS, required=True)
    single.add_argument("--supplier", choices=PARTICIPANTS, required=True)
    single.add_argument("--seed", type=int, default=42)

    batch = modes.add_parser("batch")
    _shared(batch)
    environments = batch.add_mutually_exclusive_group()
    environments.add_argument("--env", type=int, choices=range(1, 7))
    environments.add_argument(
        "--envs",
        nargs="+",
        type=int,
        choices=range(1, 7),
        metavar="ID",
    )
    batch.add_argument(
        "--contracts-root",
        type=Path,
        default=DEFAULT_CONTRACTS_ROOT,
    )
    batch.add_argument("--customers", nargs="+", choices=PARTICIPANTS, required=True)
    batch.add_argument("--suppliers", nargs="+", choices=PARTICIPANTS, required=True)
    batch.add_argument("--seeds", nargs="+", type=int, default=[42])
    batch.add_argument("--max-concurrency", type=_positive_int)
    batch.add_argument("--resume-run-dir", type=Path)
    return parser


def _contract_dirs(
    args: argparse.Namespace,
    environment_id: int,
) -> list[Path]:
    if args.mode == "single":
        contract = args.contract_dir.expanduser().resolve()
        if not (
            contract.is_dir()
            and (contract / "contract.txt").is_file()
            and (contract / "terms.json").is_file()
            and (contract / "config.json").is_file()
        ):
            raise RuntimeError(f"incomplete contract directory: {contract}")
        validate_contract_environment(contract, environment_id)
        return [contract]
    root = args.contracts_root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"contracts root is not a directory: {root}")
    candidates = sorted(
        config_path.parent
        for config_path in root.rglob("config.json")
        if (config_path.parent / "contract.txt").is_file()
        and (config_path.parent / "terms.json").is_file()
    )

    def included_in_batch(path: Path) -> bool:
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            config_path = path / "config.json"
            raise ValueError(
                f"Cannot load contract config {config_path}: {exc}"
            ) from exc
        return config.get("contingency_variant") not in EXCLUDED_BATCH_VARIANTS

    contracts = [
        path
        for path in candidates
        if contract_environment_id(path) == environment_id
        and included_in_batch(path)
    ]
    if not contracts:
        raise RuntimeError(
            f"no contracts for environment {environment_id} found in {root}"
        )
    return contracts


def _requested_environments(args: argparse.Namespace) -> list[int]:
    if args.mode == "single":
        return [args.env]
    if args.envs is not None:
        return list(args.envs)
    if args.env is not None:
        return [args.env]
    return list(DEFAULT_ENVIRONMENTS)


def run_environments(
    args: argparse.Namespace,
    environment_ids: list[int],
    contracts_by_environment: dict[int, list[Path]],
    experiment_dir: Path,
) -> int:
    """Run complete environment batches one at a time in requested order."""
    customers = [args.customer] if args.mode == "single" else args.customers
    suppliers = [args.supplier] if args.mode == "single" else args.suppliers
    seeds = [args.seed] if args.mode == "single" else args.seeds
    max_concurrency = 1 if args.mode == "single" else args.max_concurrency
    resume = (
        args.mode == "batch"
        and getattr(args, "resume_run_dir", None) is not None
    )
    stale_root = (
        experiment_dir.parent / ".resume-stale" / experiment_dir.name
        if resume
        else None
    )

    print(
        f"Campaign plan: {len(environment_ids)} environment(s) in series",
        flush=True,
    )
    for position, environment_id in enumerate(environment_ids, start=1):
        contracts = contracts_by_environment[environment_id]
        print(
            f"\n[{position}/{len(environment_ids)}] Starting environment "
            f"{environment_id} with {len(contracts)} contract(s)",
            flush=True,
        )
        result = run_batch(
            contract_dirs=contracts,
            customers=customers,
            suppliers=suppliers,
            seeds=seeds,
            environment_id=environment_id,
            goal_treatment=args.goal_treatment,
            experiment_dir=experiment_dir / f"env-{environment_id}",
            max_concurrency=max_concurrency,
            dry_run=args.dry_run,
            resume=resume,
            resume_stale_dir=(
                stale_root / f"env-{environment_id}" if stale_root else None
            ),
        )
        if result != 0:
            print(
                f"Environment {environment_id} failed; later environments "
                "were not started.",
                flush=True,
            )
            return result
        print(f"Environment {environment_id} complete.", flush=True)

    print("\nAll requested environments completed.", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    environment_ids = _requested_environments(args)
    if len(set(environment_ids)) != len(environment_ids):
        parser.error("environment IDs must not contain duplicates")
    try:
        contracts_by_environment = {
            environment_id: _contract_dirs(args, environment_id)
            for environment_id in environment_ids
        }
        category = (
            "performance-main"
            if args.goal_treatment == "control"
            else "performance-abl"
        )
        if args.mode == "batch" and args.resume_run_dir is not None:
            experiment_dir = args.resume_run_dir.expanduser().resolve()
            if not experiment_dir.is_dir():
                raise RuntimeError(
                    f"resume campaign is not a directory: {experiment_dir}"
                )
        else:
            timestamp = datetime.now().astimezone().strftime(
                "%Y-%m-%d_%H-%M-%S_%f"
            )
            experiment_dir = (
                args.results_root.expanduser().resolve() / category / timestamp
            )
            if not args.dry_run:
                experiment_dir.mkdir(parents=True, exist_ok=False)
        print(f"Experiment directory: {experiment_dir}")
        return run_environments(
            args,
            environment_ids,
            contracts_by_environment,
            experiment_dir,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
