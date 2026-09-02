#!/usr/bin/env python3
"""Run fixed-Env-5 domain-robustness negotiation campaigns."""

from __future__ import annotations

import argparse
import itertools
import json
import signal
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
REPO_ROOT = BUNDLE_ROOT.parent
for root in (REPO_ROOT, BUNDLE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from domains import DOMAINS, DOMAIN_NAMES  # noqa: E402
from llm_runtime import (  # noqa: E402
    build_runtime,
    runtime_options,
    save_domain_agent_states,
    trace_recorder,
)
from protocol import run_negotiation  # noqa: E402
from llm_utils import render_template  # noqa: E402
from llm_utils.artifacts import claim_output_directory, write_json  # noqa: E402
from llm_utils.llm import PROVIDERS  # noqa: E402
from llm_utils.process_runner import (  # noqa: E402
    ProcessJob,
    ProcessTermination,
    run_process_jobs,
    terminate_on_signals,
)
from negotiation.prompting.ablations import treatment_metadata  # noqa: E402


PROMPT_ROOT = BUNDLE_ROOT / "domain-prompts"
PROMPT_METADATA = {
    "hotel-cleaning": ("hotel-cleaning-env9-v1", "fixed_hotel_cleaning_env9_scaled100", "fixed_files"),
    "ai-model-hosting": ("ai-model-hosting-env9-v1", "fixed_ai_model_hosting_env9_scaled1000", "fixed_files"),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _environment(domain: str) -> dict[str, Any]:
    scale = DOMAINS[domain]["money_scale"]
    return {
        "domain": domain,
        "environment_id": 5,
        "environment_name": f"env-05_{domain}_scaled{scale}",
        "source_environment_id": 5,
        "source_environment_name": "env-05_baseline_current_cap20",
        "monetary_scale": scale,
        "canonical_money_rule": f"displayed_money / {scale}",
    }


def _players(
    domain: str,
    goal_treatment: str,
    max_rounds: int,
) -> list[dict[str, str]]:
    root = PROMPT_ROOT / domain / "negotiation"
    if goal_treatment != "control":
        raise ValueError("fixed domain prompts support the control treatment only")
    players = []
    for name, prompt_role, canonical_role in zip(
        DOMAINS[domain]["negotiation_names"],
        ("customer", "supplier"),
        ("customer", "caterer"),
        strict=True,
    ):
        info = (root / f"{prompt_role}_info.txt").read_text(encoding="utf-8").strip()
        goal = render_template(
            root / f"{prompt_role}_goal.txt",
            {"MAX_ROUNDS": max_rounds},
        )
        if not info or not goal:
            raise ValueError(f"empty fixed prompt for {name}")
        if "{{" in info + goal or "}}" in info + goal:
            raise ValueError(f"unresolved fixed prompt variable for {name}")
        players.append({"name": name, "canonical_role": canonical_role, "info": info, "goal": goal})
    return players


def _save_agents(output_dir: Path, domain: str, runtime) -> None:
    spec = DOMAINS[domain]
    save_domain_agent_states(
        output_dir,
        runtime.agents,
        runtime.configs,
        roles=("customer", "caterer"),
        slugs=spec["negotiation_slugs"],
    )


def _validate_output_target(output_dir: Path, *, scheduler_owned: bool) -> None:
    """Match the outside worker's stale-output protection."""
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {output_dir}")
    allowed = {"stdout.log"} if scheduler_owned else set()
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise RuntimeError(
            "output directory is not empty; choose a new run directory: "
            f"{output_dir}"
        )
    if scheduler_owned:
        log_path = output_dir / "stdout.log"
        try:
            mode = log_path.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise RuntimeError(
                f"scheduler-owned output is missing stdout.log: {output_dir}"
            ) from exc
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise RuntimeError(
                "scheduler-owned stdout.log must be a private regular file: "
                f"{log_path}"
            )


def _worker(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    _validate_output_target(output_dir, scheduler_owned=args.scheduler_owned_output)
    providers = (args.first_provider, args.second_provider)
    players = _players(args.domain, args.goal_treatment, args.max_rounds)
    log_reasoning = not args.no_log_reasoning or args.debug_json
    options = runtime_options(output_dir)
    runtime = build_runtime(
        players,
        providers,
        output_dir=output_dir,
        log_reasoning=log_reasoning,
        options=options,
    )
    claim = claim_output_directory(
        output_dir,
        metadata={
            "implementation": f"domain-robustness-negotiation-{args.domain}",
            "environment_id": 5,
            "domain": args.domain,
        },
    )
    with claim:
        recorder = trace_recorder(output_dir, log_reasoning)
        recorder.reset()

        data_dir = output_dir / "data"
        config_dir = output_dir / "config"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        write_json(config_dir / "negotiation_players.json", players)

        prompt_version, prompt_profile, prompt_source = PROMPT_METADATA[args.domain]
        run_metadata = {
            **_environment(args.domain),
            "implementation": f"domain-robustness-negotiation-{args.domain}",
            "implementation_version": 2,
            "customer_provider": providers[0],
            "customer_model": runtime.configs[0].model,
            "customer_reasoning": runtime.configs[0].reasoning_metadata,
            "caterer_provider": providers[1],
            "caterer_model": runtime.configs[1].model,
            "caterer_reasoning": runtime.configs[1].reasoning_metadata,
            "max_rounds": args.max_rounds,
            "prompt_layout": "chat_v2",
            "prompt_template_version": prompt_version,
            "prompt_content_profile": prompt_profile,
            "prompt_source": prompt_source,
            "prompt_cache_enabled": options.prompt_cache_enabled,
            "thought_traces_enabled": True,
            "empty_response_policy": "retry_once_then_invalid",
            **treatment_metadata(args.goal_treatment),
            "output_dir": str(output_dir),
        }
        write_json(config_dir / "run_config.json", run_metadata)
        contract, accepted, status = run_negotiation(
            args.domain,
            runtime.agents,
            runtime.models,
            max_rounds=args.max_rounds,
            log_path=data_dir / "negotiation_log.txt",
            contract_path=data_dir / "contract.txt",
            debug_json=args.debug_json,
            record_llm_call=recorder.record,
        )
        _save_agents(output_dir, args.domain, runtime)
        write_json(data_dir / "summary.json", {
            **run_metadata,
            **status,
            "accepted": accepted,
            "contract_text": contract or "",
        })
        claim.finalize(
            "complete" if status["valid"] else "invalid",
            {**status, "accepted": accepted, "environment_id": 5},
        )
    return 0


def _worker_command(args, first_provider: str, second_provider: str, output_dir: Path) -> tuple[str, ...]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--domain", args.domain,
        "--first-provider", first_provider,
        "--second-provider", second_provider,
        "--output-dir", str(output_dir),
        "--max-rounds", str(args.max_rounds),
        "--goal-treatment", args.goal_treatment,
        "--scheduler-owned-output",
    ]
    if args.debug_json:
        command.append("--debug-json")
    if args.no_log_reasoning:
        command.append("--no-log-reasoning")
    return tuple(command)


def _jobs(args: argparse.Namespace) -> list[ProcessJob]:
    pairings = [tuple(args.match)] if args.match else list(itertools.product(args.providers, repeat=2))
    experiment_dir = args.out_root.resolve() / f"env 5 - {datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
    return [
        ProcessJob(
            label=f"run {run_number}/{first}_vs_{second}",
            argv=_worker_command(
                args,
                first,
                second,
                experiment_dir / f"run {run_number}" / f"{first}_vs_{second}",
            ),
            output_dir=experiment_dir / f"run {run_number}" / f"{first}_vs_{second}",
            log_path=experiment_dir / f"run {run_number}" / f"{first}_vs_{second}" / "stdout.log",
            cwd=BUNDLE_ROOT,
        )
        for run_number in range(1, args.runs + 1)
        for first, second in pairings
    ]


def _run_batch(args: argparse.Namespace) -> int:
    _players(args.domain, args.goal_treatment, args.max_rounds)
    jobs = _jobs(args)
    concurrency = min(args.max_concurrency or len(jobs), len(jobs))
    print(f"{args.domain}: {len(jobs)} negotiation(s); concurrency={concurrency}")
    if args.dry_run:
        for job in jobs:
            print(f"  [{job.label}] {' '.join(job.argv)}")
        return 0

    failed: list[str] = []
    invalid: list[str] = []

    def job_status(result) -> tuple[str, str | None]:
        if result.returncode != 0:
            return f"FAILED (exit {result.returncode})", "failed"
        summary_path = result.job.output_dir / "data" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return f"FAILED (invalid summary: {exc})", "failed"
        if not isinstance(summary, dict):
            return "FAILED (invalid summary: expected a JSON object)", "failed"
        if type(summary.get("valid")) is not bool:
            return "FAILED (invalid summary: valid must be a boolean)", "failed"
        if type(summary.get("accepted")) is not bool:
            return "FAILED (invalid summary: accepted must be a boolean)", "failed"
        if summary["accepted"]:
            contract = summary.get("contract_text")
            if not isinstance(contract, str) or not contract.strip():
                return "FAILED (accepted run requires nonempty contract_text)", "failed"
        turns = summary.get("turns_taken")
        if type(turns) is not int or turns < 0:
            return "FAILED (turns_taken must be a nonnegative integer)", "failed"
        if not summary["valid"]:
            reason = summary.get("invalid_reason")
            if not isinstance(reason, str) or not reason.strip():
                return "FAILED (invalid run requires invalid_reason)", "failed"
            return f"INVALID ({reason})", "invalid"
        return "OK", None

    def complete(result) -> None:
        status, category = job_status(result)
        if category == "failed":
            failed.append(result.job.label)
        elif category == "invalid":
            invalid.append(result.job.label)
        print(f"  [{result.job.label}] {status}", flush=True)

    try:
        run_process_jobs(
            jobs,
            max_concurrency=concurrency,
            on_start=lambda job, pid: print(f"  [{job.label}] started (PID {pid})", flush=True),
            on_complete=complete,
        )
    except ProcessTermination as exc:
        print(
            f"Batch received {signal.Signals(exc.signum).name}; "
            "all active workers were stopped."
        )
        return exc.exit_code
    except KeyboardInterrupt:
        print("Batch interrupted; all active workers were stopped.")
        return 130
    if failed or invalid:
        return 1
    print(f"Negotiation experiment saved to: {jobs[0].output_dir.parents[1]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAIN_NAMES, required=True)
    parser.add_argument("--env", type=int, choices=(5,), default=5)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--match", nargs=2, choices=PROVIDERS)
    parser.add_argument("--runs", type=_positive_int, default=1)
    parser.add_argument("--max-rounds", type=_positive_int, default=50)
    parser.add_argument("--max-concurrency", type=_positive_int)
    parser.add_argument("--goal-treatment", choices=("control",), default="control")
    parser.add_argument("--out-root", type=Path, default=BUNDLE_ROOT / "runs" / "negotiation")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-json", action="store_true")
    parser.add_argument("--no-log-reasoning", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--first-provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument("--second-provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scheduler-owned-output", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.providers = list(dict.fromkeys(args.providers))
    if args.worker:
        if args.first_provider is None or args.second_provider is None or args.output_dir is None:
            parser.error("worker arguments are incomplete")
        try:
            with terminate_on_signals():
                return _worker(args)
        except ProcessTermination as exc:
            return exc.exit_code
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    try:
        players = _players(args.domain, args.goal_treatment, args.max_rounds)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(f"Fixed Env 5 {args.domain} negotiation prompts: OK ({len(players)} roles)")
        return 0
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
