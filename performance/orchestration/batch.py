"""Build and run performance worker jobs."""

from __future__ import annotations

import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from llm_utils.process_runner import (
    ProcessJob,
    ProcessResult,
    ProcessTermination,
    run_process_jobs,
)
from performance.contracts import contract_id


ORCHESTRATION_DIR = Path(__file__).resolve().parent
WORKER = ORCHESTRATION_DIR / "worker.py"


@dataclass(frozen=True)
class PerformanceJob:
    contract_id: str
    contract_dir: Path
    customer: str
    supplier: str
    seed: int
    output_dir: Path
    command: tuple[str, ...]

    @property
    def label(self) -> str:
        return (
            f"{self.contract_id}/customer-{self.customer}_"
            f"supplier-{self.supplier}_seed-{self.seed}"
        )

    def process_job(self) -> ProcessJob:
        return ProcessJob(
            label=self.label,
            argv=self.command,
            output_dir=self.output_dir,
            log_path=self.output_dir / "stdout.log",
            cwd=ORCHESTRATION_DIR,
        )


def build_jobs(
    *,
    contract_dirs: list[Path],
    customers: list[str],
    suppliers: list[str],
    seeds: list[int],
    environment_id: int,
    goal_treatment: str,
    experiment_dir: Path,
) -> list[PerformanceJob]:
    jobs: list[PerformanceJob] = []
    for contract_dir in contract_dirs:
        stable_contract_id = contract_id(contract_dir)
        for customer in customers:
            for supplier in suppliers:
                for seed in seeds:
                    run_name = (
                        f"customer-{customer}_supplier-{supplier}_seed-{seed}"
                    )
                    output_dir = experiment_dir / stable_contract_id / run_name
                    command = (
                        sys.executable,
                        str(WORKER),
                        "--env",
                        str(environment_id),
                        "--contract-dir",
                        str(contract_dir),
                        "--customer",
                        customer,
                        "--supplier",
                        supplier,
                        "--seed",
                        str(seed),
                        "--goal-treatment",
                        goal_treatment,
                        "--output-dir",
                        str(output_dir),
                    )
                    jobs.append(
                        PerformanceJob(
                            contract_id=stable_contract_id,
                            contract_dir=contract_dir,
                            customer=customer,
                            supplier=supplier,
                            seed=seed,
                            output_dir=output_dir,
                            command=command,
                        )
                    )
    return jobs


def _job_status(job: PerformanceJob, returncode: int) -> tuple[str, bool]:
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
        return f"INVALID ({summary.get('invalid_reason', 'unknown')})", False
    return "OK", True


def _completed_for_resume(job: PerformanceJob) -> bool:
    status_path = job.output_dir / "run_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        isinstance(status, dict)
        and status.get("status") == "complete"
        and _job_status(job, 0)[1]
    )


def _archive_incomplete_output(
    job: PerformanceJob,
    *,
    experiment_dir: Path,
    stale_dir: Path,
) -> None:
    if not job.output_dir.exists():
        return
    relative = job.output_dir.relative_to(experiment_dir)
    archive = stale_dir / relative
    if archive.exists():
        archive = archive.with_name(f"{archive.name}-{time.time_ns()}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.rename(archive)
    print(f"  [{job.label}] archived incomplete output to {archive}")


def run_batch(
    *,
    contract_dirs: list[Path],
    customers: list[str],
    suppliers: list[str],
    seeds: list[int],
    environment_id: int,
    goal_treatment: str,
    experiment_dir: Path,
    max_concurrency: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    resume_stale_dir: Path | None = None,
) -> int:
    all_jobs = build_jobs(
        contract_dirs=contract_dirs,
        customers=customers,
        suppliers=suppliers,
        seeds=seeds,
        environment_id=environment_id,
        goal_treatment=goal_treatment,
        experiment_dir=experiment_dir,
    )
    if not all_jobs:
        raise ValueError("at least one performance job is required")
    jobs = all_jobs
    if resume:
        if resume_stale_dir is None:
            raise ValueError("resume requires a stale-output directory")
        jobs = [job for job in all_jobs if not _completed_for_resume(job)]
        print(
            f"Resume: skipped {len(all_jobs) - len(jobs)} completed job(s); "
            f"{len(jobs)} remaining"
        )
        if not dry_run:
            for job in jobs:
                _archive_incomplete_output(
                    job,
                    experiment_dir=experiment_dir,
                    stale_dir=resume_stale_dir,
                )
    if not jobs:
        return 0
    concurrency = min(max_concurrency or len(jobs), len(jobs))
    print(f"Performance jobs: {len(jobs)}; concurrency={concurrency}")

    if dry_run:
        for job in jobs:
            print(f"  [{job.label}] {' '.join(job.command)}")
        return 0

    failed: list[str] = []

    def complete(result: ProcessResult) -> None:
        job = next(job for job in jobs if job.output_dir == result.job.output_dir)
        status, succeeded = _job_status(job, result.returncode)
        if not succeeded:
            failed.append(job.label)
        print(f"  [{job.label}] {status}", flush=True)

    try:
        run_process_jobs(
            (job.process_job() for job in jobs),
            max_concurrency=concurrency,
            on_start=lambda job, pid: print(
                f"  [{job.label}] started (PID {pid})",
                flush=True,
            ),
            on_complete=complete,
        )
    except ProcessTermination as exc:
        print(f"Batch stopped by {signal.Signals(exc.signum).name}")
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    return 1 if failed else 0
