#!/usr/bin/env python3
"""Run fixed-Env-5 domain-robustness performance campaigns."""

from __future__ import annotations

import argparse
import itertools
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
REPO_ROOT = BUNDLE_ROOT.parent
for root in (REPO_ROOT, BUNDLE_ROOT, HERE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from contracts import index as contract_index, select as select_contracts  # noqa: E402
from domain import load_contract_views  # noqa: E402
from domains import DOMAINS, DOMAIN_NAMES  # noqa: E402
from prompts import validate as validate_prompts  # noqa: E402
from worker import run_one  # noqa: E402
from llm_utils.llm import PROVIDERS  # noqa: E402
from llm_utils.process_runner import (  # noqa: E402
    ProcessJob,
    ProcessTermination,
    run_process_jobs,
    terminate_on_signals,
)


DEFAULT_CONTRACTS = REPO_ROOT / "algorithms" / "synth-contracts" / "data" / "suite" / "env-05"
BASELINES = ("rc", "re", "rcc")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _roles(domain: str) -> tuple[str, str]:
    return DOMAINS[domain]["performance_roles"]


def _selected_contracts(args: argparse.Namespace) -> list[str]:
    selected = select_contracts(
        args.contracts_root, args.contract_ids, args.contract_prefix, args.limit
    )
    directories = contract_index(args.contracts_root)
    for stable_id in selected:
        load_contract_views(directories[stable_id], args.domain)
    return selected


def _run_name(domain: str, role: str, provider: str, baseline: str, seed: int) -> str:
    focal_customer = role == _roles(domain)[0]
    customer = provider if focal_customer else baseline
    supplier = baseline if focal_customer else provider
    return f"customer-{customer}_supplier-{supplier}_seed-{seed}"


def _worker_command(args, contract: str, provider: str, baseline: str, seed: int, output: Path) -> tuple[str, ...]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--domain", args.domain,
        "--role", args.role,
        "--contracts-root", str(args.contracts_root),
        "--contract-id", contract,
        "--provider", provider,
        "--baseline", baseline,
        "--seed", str(seed),
        "--output-dir", str(output),
        "--goal-treatment", args.goal_treatment,
    ]
    if args.no_log_reasoning:
        command.append("--no-log-reasoning")
    return tuple(command)


def _worker(args: argparse.Namespace) -> int:
    run_one(
        domain=args.domain,
        role=args.role,
        contracts_root=args.contracts_root,
        contract_id=args.contract_id,
        provider=args.provider,
        baseline_name=args.baseline,
        seed=args.seed,
        out_dir=args.output_dir,
        log_reasoning=not args.no_log_reasoning,
        goal_treatment=args.goal_treatment,
    )
    return 0


def _experiment_dir(args: argparse.Namespace) -> Path:
    configured = args.out_root or args.out_dir
    if configured is not None:
        return configured.resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return (
        BUNDLE_ROOT / "runs" / "performance" / args.domain /
        args.role.replace("-", "_") / f"env 5 - {stamp}"
    )


def _jobs(args, contracts: list[str], baselines: list[str]) -> tuple[Path, list[ProcessJob]]:
    experiment = _experiment_dir(args)
    jobs = []
    for contract, provider, baseline, seed in itertools.product(
        contracts, args.providers, baselines, args.seeds
    ):
        output = experiment / contract / _run_name(
            args.domain, args.role, provider, baseline, seed
        )
        label = f"{contract} | role={args.role} | provider={provider} | baseline={baseline} | seed={seed}"
        jobs.append(ProcessJob(
            label=label,
            argv=_worker_command(args, contract, provider, baseline, seed, output),
            output_dir=output,
            log_path=output / "stdout.log",
            cwd=BUNDLE_ROOT,
        ))
    return experiment, jobs


def _job_status(job: ProcessJob, returncode: int) -> tuple[str, bool]:
    if returncode != 0:
        return f"FAILED (exit {returncode})", False
    path = job.output_dir / "data" / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"FAILED (invalid summary: {exc})", False
    if not isinstance(summary, dict) or type(summary.get("valid")) is not bool:
        return "FAILED (summary requires boolean valid)", False
    if not summary["valid"]:
        reason = summary.get("invalid_reason", "unknown")
        return f"INVALID ({reason})", False
    return "OK", True


def _completed_for_resume(job: ProcessJob) -> bool:
    try:
        status = json.loads(
            (job.output_dir / "run_status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        isinstance(status, dict)
        and status.get("status") == "complete"
        and _job_status(job, 0)[1]
    )


def _archive_incomplete_output(
    job: ProcessJob,
    *,
    experiment: Path,
    stale_root: Path,
) -> None:
    if not job.output_dir.exists():
        return
    archive = stale_root / job.output_dir.relative_to(experiment)
    if archive.exists():
        archive = archive.with_name(f"{archive.name}-{time.time_ns()}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.rename(archive)
    print(f"  [{job.label}] archived incomplete output to {archive}")


def _run_batch(args, contracts: list[str], baselines: list[str]) -> int:
    experiment, all_jobs = _jobs(args, contracts, baselines)
    jobs = all_jobs
    if args.skip_existing:
        jobs = [job for job in all_jobs if not _completed_for_resume(job)]
        print(
            f"Resume: skipped {len(all_jobs) - len(jobs)} completed job(s); "
            f"{len(jobs)} remaining"
        )
        if not args.dry_run:
            stale_root = experiment / ".stale"
            for job in jobs:
                _archive_incomplete_output(
                    job,
                    experiment=experiment,
                    stale_root=stale_root,
                )
    print("Environment: fixed Env 5")
    print(f"Domain: {args.domain}")
    print(f"Role: {args.role}")
    print(f"Contracts root: {args.contracts_root}")
    print(f"Runs: {len(jobs)}")
    print(f"Output root: {experiment}")
    if args.dry_run:
        for job in jobs[:20]:
            print(f"  [{job.label}]")
        if len(jobs) > 20:
            print(f"  ... {len(jobs) - 20} more")
        return 0
    if not jobs:
        print("Nothing to run.")
        return 0

    failures: list[str] = []
    invalid: list[str] = []

    def complete(result: Any) -> None:
        status, succeeded = _job_status(result.job, result.returncode)
        if not succeeded:
            if status.startswith("INVALID"):
                invalid.append(result.job.label)
            else:
                failures.append(result.job.label)
        print(f"  [{result.job.label}] {status}", flush=True)

    try:
        run_process_jobs(
            jobs,
            max_concurrency=min(args.max_parallel, len(jobs)),
            on_start=lambda job, pid: print(f"  [{job.label}] started (PID {pid})", flush=True),
            on_complete=complete,
        )
    except ProcessTermination as exc:
        print(f"Batch stopped by {signal.Signals(exc.signum).name}")
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    return 1 if failures or invalid else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAIN_NAMES, required=True, help=argparse.SUPPRESS)
    parser.add_argument("--role", metavar="ROLE", help="LLM-controlled domain role")
    parser.add_argument("--env", type=int, choices=(5,), default=5)
    parser.add_argument("--contracts-root", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--contract-ids", nargs="+")
    parser.add_argument("--contract-prefix")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--max-parallel", "--max-concurrency", dest="max_parallel", type=_positive_int, default=4)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--goal-treatment", choices=("control",), default="control")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log-reasoning", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--contract-id", help=argparse.SUPPRESS)
    parser.add_argument("--provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", choices=BASELINES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    args.contracts_root = args.contracts_root.expanduser().resolve()
    args.providers = list(dict.fromkeys(args.providers))
    roles = _roles(args.domain)
    if args.role is not None and args.role not in roles:
        parser.error(f"--role for {args.domain} must be one of: {', '.join(roles)}")
    if args.worker:
        required = (args.role, args.contract_id, args.provider, args.baseline, args.seed, args.output_dir)
        if any(value is None for value in required):
            parser.error("worker arguments are incomplete")
        try:
            with terminate_on_signals():
                return _worker(args)
        except ProcessTermination as exc:
            return exc.exit_code
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    try:
        validate_prompts(args.domain)
        contracts = _selected_contracts(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(f"Fixed Env 5 {args.domain} performance prompts and adapter: OK ({len(contracts)} contract(s))")
        return 0
    if args.role is None:
        parser.error("--role is required unless --validate-only is used")
    return _run_batch(
        args,
        contracts,
        list(dict.fromkeys(args.baselines or BASELINES)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
