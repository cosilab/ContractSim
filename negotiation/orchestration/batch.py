"""Plan and evaluate one environment's negotiation worker jobs."""

from __future__ import annotations

import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_utils.process_runner import (
    ProcessJob,
    ProcessResult,
    ProcessTermination,
    run_process_jobs,
)


ORCHESTRATION_DIR = Path(__file__).resolve().parent
WORKER = ORCHESTRATION_DIR / "worker.py"


@dataclass(frozen=True)
class NegotiationJob:
    """Negotiation-specific identity and command for one worker."""

    run_number: int
    customer_provider: str
    caterer_provider: str
    output_dir: Path
    command: tuple[str, ...]

    @property
    def label(self) -> str:
        pairing = f"{self.customer_provider}_vs_{self.caterer_provider}"
        return f"run {self.run_number}/{pairing}"

    def process_job(self) -> ProcessJob:
        """Project this phase-owned job onto the shared process boundary."""
        return ProcessJob(
            label=self.label,
            argv=self.command,
            output_dir=self.output_dir,
            log_path=self.output_dir / "stdout.log",
            cwd=ORCHESTRATION_DIR,
        )


def _worker_command(
    *,
    environment_id: int,
    customer_provider: str,
    caterer_provider: str,
    output_dir: Path,
    max_rounds: int,
    goal_treatment: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(WORKER),
        "--env",
        str(environment_id),
        "--customer-provider",
        customer_provider,
        "--caterer-provider",
        caterer_provider,
        "--output-dir",
        str(output_dir),
        "--max-rounds",
        str(max_rounds),
        "--goal-treatment",
        goal_treatment,
        "--scheduler-owned-output",
    ]
    return tuple(command)


def build_jobs(
    *,
    environment_id: int,
    pairings: list[tuple[str, str]],
    runs: int,
    environment_dir: Path,
    max_rounds: int,
    goal_treatment: str,
) -> list[NegotiationJob]:
    jobs = []
    for run_number in range(1, runs + 1):
        run_dir = environment_dir / f"run-{run_number}"
        for customer_provider, caterer_provider in pairings:
            output_dir = run_dir / f"{customer_provider}_vs_{caterer_provider}"
            jobs.append(
                NegotiationJob(
                    run_number=run_number,
                    customer_provider=customer_provider,
                    caterer_provider=caterer_provider,
                    output_dir=output_dir,
                    command=_worker_command(
                        environment_id=environment_id,
                        customer_provider=customer_provider,
                        caterer_provider=caterer_provider,
                        output_dir=output_dir,
                        max_rounds=max_rounds,
                        goal_treatment=goal_treatment,
                    ),
                )
            )
    return jobs


def _job_status(job: NegotiationJob, returncode: int) -> tuple[str, str | None]:
    """Apply negotiation summary semantics to one completed process."""
    if returncode != 0:
        return f"FAILED (exit {returncode})", "failed"
    summary_path = job.output_dir / "data" / "summary.json"
    if not summary_path.is_file():
        return "FAILED (summary missing)", "failed"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return f"FAILED (invalid summary: {exc})", "failed"
    if not isinstance(summary, dict):
        return "FAILED (invalid summary: expected a JSON object)", "failed"
    if type(summary.get("valid")) is not bool:
        return "FAILED (invalid summary: valid must be a boolean)", "failed"
    if type(summary.get("accepted")) is not bool:
        return "FAILED (invalid summary: accepted must be a boolean)", "failed"
    if summary["accepted"]:
        contract_text = summary.get("contract_text")
        if not isinstance(contract_text, str) or not contract_text.strip():
            return (
                "FAILED (invalid summary: accepted runs require nonempty "
                "contract_text)",
                "failed",
            )
    turns_taken = summary.get("turns_taken")
    if type(turns_taken) is not int or turns_taken < 0:
        return (
            "FAILED (invalid summary: turns_taken must be a nonnegative integer)",
            "failed",
        )
    if not summary["valid"]:
        reason = summary.get("invalid_reason")
        if not isinstance(reason, str) or not reason.strip():
            return (
                "FAILED (invalid summary: invalid_reason must be a nonempty "
                "string when valid is false)",
                "failed",
            )
        return f"INVALID ({reason})", "invalid"
    return "OK", None


def _run_jobs(
    jobs: list[NegotiationJob],
    max_concurrency: int,
) -> tuple[list[str], list[str]]:
    """Run generic processes while retaining phase-specific classification."""
    failed: list[str] = []
    invalid: list[str] = []
    jobs_by_output = {job.output_dir: job for job in jobs}

    def report_start(job: ProcessJob, pid: int) -> None:
        print(f"  [{job.label}] started (PID {pid}) -> {job.output_dir}", flush=True)

    def classify(result: ProcessResult) -> None:
        job = jobs_by_output[result.job.output_dir]
        status, category = _job_status(job, result.returncode)
        if category == "failed":
            failed.append(job.label)
        elif category == "invalid":
            invalid.append(job.label)
        print(f"  [{job.label}] {status}", flush=True)

    run_process_jobs(
        (job.process_job() for job in jobs),
        max_concurrency=max_concurrency,
        on_start=report_start,
        on_complete=classify,
    )
    return failed, invalid


def run_batch(
    *,
    environment_id: int,
    pairings: list[tuple[str, str]],
    runs: int,
    max_rounds: int,
    goal_treatment: str,
    experiment_dir: Path,
    max_concurrency: int | None = None,
    dry_run: bool = False,
) -> int:
    if environment_id not in range(1, 7):
        raise ValueError("environment_id must be between 1 and 6")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if not pairings:
        raise ValueError("at least one provider pairing is required")
    if len(set(pairings)) != len(pairings):
        raise ValueError("provider pairings must not contain duplicates")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    environment_dir = experiment_dir / f"env-{environment_id}"
    jobs = build_jobs(
        environment_id=environment_id,
        pairings=pairings,
        runs=runs,
        environment_dir=environment_dir,
        max_rounds=max_rounds,
        goal_treatment=goal_treatment,
    )
    concurrency = min(max_concurrency or len(jobs), len(jobs))
    print(
        f"Environment {environment_id}: {runs} run(s) x {len(pairings)} pairing(s) "
        f"= {len(jobs)} negotiation(s); concurrency={concurrency}",
        flush=True,
    )
    print(f"Environment directory: {environment_dir}", flush=True)

    if dry_run:
        for job in jobs:
            print(f"  [{job.label}] {' '.join(job.command)}", flush=True)
        print("Dry run complete; no directories or workers were created.", flush=True)
        return 0

    environment_dir.mkdir(parents=True, exist_ok=False)

    try:
        failed, invalid = _run_jobs(jobs, concurrency)
    except ProcessTermination as exc:
        signal_name = signal.Signals(exc.signum).name
        print(
            f"\nBatch received {signal_name}; all active workers were stopped.",
            flush=True,
        )
        return exc.exit_code
    except KeyboardInterrupt:
        print("\nBatch interrupted; all active workers were stopped.", flush=True)
        return 130
    except OSError as exc:
        raise RuntimeError(f"Unable to launch negotiation worker: {exc}") from exc

    if failed:
        print(f"{len(failed)} negotiation(s) failed: {', '.join(failed)}")
    if invalid:
        print(f"{len(invalid)} negotiation(s) were invalid: {', '.join(invalid)}")
    return 1 if failed or invalid else 0
