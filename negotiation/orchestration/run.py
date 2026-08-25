"""Public entrypoint for single-pairing and batch negotiation campaigns."""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys
from datetime import datetime
from pathlib import Path

ORCHESTRATION_DIR = Path(__file__).resolve().parent
NEGOTIATION_DIR = ORCHESTRATION_DIR.parent
PROJECT_ROOT = NEGOTIATION_DIR.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_utils.artifacts import write_text  # noqa: E402
from llm_utils.llm import PROVIDERS  # noqa: E402
from negotiation.orchestration.batch import run_batch  # noqa: E402
from negotiation.prompting import GOAL_TREATMENT_CHOICES  # noqa: E402


GOAL_TREATMENTS = GOAL_TREATMENT_CHOICES
DEFAULT_ENVIRONMENTS = (1, 2, 3, 4, 5, 6)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--envs",
        nargs="+",
        type=int,
        choices=range(1, 7),
        default=list(DEFAULT_ENVIRONMENTS),
        metavar="ID",
        help="environment IDs to process in series (default: 1 2 3 4 5 6)",
    )
    parser.add_argument("--max-rounds", type=positive_int, default=50)
    parser.add_argument(
        "--goal-treatment",
        choices=GOAL_TREATMENTS,
        default="control",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="repository results directory (default: ./results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the execution plan without validating prompts or launching workers",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    single = modes.add_parser(
        "single",
        help="run one selected provider pairing in every requested environment",
    )
    _add_shared_arguments(single)
    single.add_argument(
        "--match",
        nargs=2,
        choices=PROVIDERS,
        required=True,
        metavar=("CUSTOMER", "CATERER"),
    )

    batch = modes.add_parser(
        "batch",
        help="run provider matrices and replicates in every requested environment",
    )
    _add_shared_arguments(batch)
    batch.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    batch.add_argument(
        "--runs",
        type=positive_int,
        default=1,
        metavar="N",
        help="independent provider matrices per environment (default: 1)",
    )
    batch.add_argument(
        "--max-concurrency",
        type=positive_int,
        metavar="N",
        help=(
            "maximum simultaneous negotiations within one environment "
            "(default: all jobs for that environment)"
        ),
    )
    return parser


def _prepare_experiment(args: argparse.Namespace) -> Path:
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y-%m-%d_%H-%M-%S_%f")
    category = (
        "negotiation-main"
        if args.goal_treatment == "control"
        else "negotiation-abl"
    )
    experiment_dir = args.results_root / category / timestamp
    args.experiment_dir = experiment_dir
    print(f"Experiment directory: {experiment_dir}", flush=True)
    if args.dry_run:
        return experiment_dir

    try:
        experiment_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to create experiment directory {experiment_dir}: {exc}"
        ) from exc
    command = shlex.join([sys.executable, *sys.argv])
    record = (
        f"started-at: {started_at.isoformat()}\n"
        f"command: {command}\n"
        f"working-directory: {Path.cwd().resolve()}\n"
        f"category: {category}\n"
        f"mode: {args.mode}\n"
        f"goal-treatment: {args.goal_treatment}\n"
        f"environments: {' '.join(str(item) for item in args.envs)}\n"
    )
    try:
        write_text(experiment_dir / "run.txt", record)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to write experiment record in {experiment_dir}: {exc}"
        ) from exc
    return experiment_dir


def _pairings_for(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.mode == "single":
        return [(args.match[0], args.match[1])]
    return list(itertools.product(args.providers, repeat=2))


def run_environments(args: argparse.Namespace) -> int:
    """Process environments in order, waiting for each batch to finish."""
    pairings = _pairings_for(args)
    runs = 1 if args.mode == "single" else args.runs
    max_concurrency = 1 if args.mode == "single" else args.max_concurrency
    jobs_per_environment = runs * len(pairings)

    print(
        f"Campaign plan: {len(args.envs)} environment(s) in series; "
        f"{jobs_per_environment} negotiation(s) per environment; mode={args.mode}",
        flush=True,
    )
    for position, environment_id in enumerate(args.envs, start=1):
        print(
            f"\n[{position}/{len(args.envs)}] Starting environment {environment_id}",
            flush=True,
        )
        result = run_batch(
            environment_id=environment_id,
            pairings=pairings,
            runs=runs,
            max_rounds=args.max_rounds,
            goal_treatment=args.goal_treatment,
            experiment_dir=args.experiment_dir,
            max_concurrency=max_concurrency,
            dry_run=args.dry_run,
        )
        if result != 0:
            print(
                f"Environment {environment_id} failed; later environments were not started.",
                flush=True,
            )
            return result
        print(f"Environment {environment_id} complete.", flush=True)

    print("\nAll requested environments completed.", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if len(set(args.envs)) != len(args.envs):
        parser.error("--envs must not contain duplicates")
    if args.mode == "batch" and len(set(args.providers)) != len(args.providers):
        parser.error("--providers must not contain duplicates")
    args.results_root = args.results_root.expanduser().resolve()
    try:
        _prepare_experiment(args)
        return run_environments(args)
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
