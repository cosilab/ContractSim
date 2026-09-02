#!/usr/bin/env python3
"""Run the consolidated domain-robustness experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
PROJECT_ROOT = HERE.parent
CONFIG_PATH = HERE / "config" / "experiment.json"
ENVIRONMENT_ID = 5
PROVIDERS = ("gemini", "openai", "anthropic")
BASELINES = ("rc", "re", "rcc")
DOMAINS = ("hotel-cleaning", "ai-model-hosting")


def _load_config() -> dict[str, object]:
    payload = json.loads(CONFIG_PATH.read_text())
    if int(payload["environment_id"]) != ENVIRONMENT_ID:
        raise ValueError("domain-robustness is fixed to Env 5")
    return payload


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _domain_path(domain: str, phase: str) -> Path:
    return REPO_ROOT / phase / "run.py"


def _validate_repo() -> None:
    required = (
        REPO_ROOT / "negotiation" / "run.py",
        REPO_ROOT / "performance" / "run.py",
        PROJECT_ROOT / "envs" / "5" / "environment.json",
        PROJECT_ROOT / "algorithms" / "synth-contracts" / "data" / "suite" / "env-05",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing repository requirements:\n" + "\n".join(missing))


def _append_reasoning(command: list[str], no_log_reasoning: bool) -> None:
    if no_log_reasoning:
        command.append("--no-log-reasoning")


def _negotiation_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    for domain in args.domains:
        command = [
            sys.executable,
            str(_domain_path(domain, "negotiation")),
            "--domain",
            domain,
        ]
        command.extend(
            (
                "--providers",
                *args.providers,
                "--runs",
                str(args.negotiation_runs),
                "--max-rounds",
                str(args.max_rounds),
                "--out-root",
                str(args.out_root / "negotiation" / domain),
            )
        )
        _append_reasoning(command, args.no_log_reasoning)
        commands.append(command)
    return commands


def _performance_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = []
    for domain in args.domains:
        for role in {
            "hotel-cleaning": ("hotel-owner", "cleaning-provider"),
            "ai-model-hosting": ("ai-startup-founder", "ai-service-provider"),
        }[domain]:
            command = [
                sys.executable,
                str(_domain_path(domain, "performance")),
                "--domain",
                domain,
                "--role",
                role,
                "--contracts-root",
                str(args.contracts_root),
                "--providers",
                *args.providers,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--max-parallel",
                str(args.max_parallel),
            ]
            command.extend(
                (
                    "--baselines",
                    *args.baselines,
                    "--out-root",
                    str(args.out_root / "performance" / domain / role),
                )
            )
            command.append("--skip-existing")
            if args.contract_ids:
                command.extend(("--contract-ids", *args.contract_ids))
            if args.contract_prefix:
                command.extend(("--contract-prefix", args.contract_prefix))
            if args.limit is not None:
                command.extend(("--limit", str(args.limit)))
            _append_reasoning(command, args.no_log_reasoning)
            if args.dry_run:
                command.append("--dry-run")
            commands.append(command)
    return commands


def _commands(args: argparse.Namespace, phase: str) -> list[list[str]]:
    if phase == "negotiation":
        return _negotiation_commands(args)
    if phase == "performance":
        return _performance_commands(args)
    return _negotiation_commands(args) + _performance_commands(args)


def _render(command: list[str]) -> str:
    return " ".join(repr(part) if " " in part else part for part in command)


def _write_manifest(args: argparse.Namespace, phase: str, commands: list[list[str]]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": args.env,
        "phase": phase,
        "domains": args.domains,
        "providers": args.providers,
        "baselines": args.baselines,
        "seeds": args.seeds,
        "commands": commands,
        "contracts_root": str(args.contracts_root),
        "output_root": str(args.out_root),
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + _render(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    config = _load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "run"))
    parser.add_argument(
        "phase",
        nargs="?",
        choices=("negotiation", "performance", "all"),
        help="phase for plan/run; validate always checks both phases",
    )
    parser.add_argument("--env", type=int, default=int(config["environment_id"]), choices=(5,))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(config["domains"]))
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(config["providers"]))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(config["baselines"]))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config["performance_seeds"]))
    parser.add_argument("--negotiation-runs", type=_positive, default=int(config["negotiation_runs"]))
    parser.add_argument("--max-rounds", type=_positive, default=int(config["negotiation_max_rounds"]))
    parser.add_argument("--max-parallel", type=_positive, default=int(config["performance_max_parallel"]))
    parser.add_argument("--contracts-root", type=Path, default=REPO_ROOT / str(config["contract_root"]))
    parser.add_argument("--contract-ids", nargs="+")
    parser.add_argument("--contract-prefix")
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--out-root", type=Path, default=HERE / "runs")
    parser.add_argument("--no-log-reasoning", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    _validate_repo()
    checks = []
    for domain in args.domains:
        checks.append(
            [
                sys.executable,
                str(_domain_path(domain, "negotiation")),
                "--domain",
                domain,
                "--validate-only",
                "--max-rounds",
                str(args.max_rounds),
            ]
        )
        checks.append(
            [
                sys.executable,
                str(_domain_path(domain, "performance")),
                "--domain",
                domain,
                "--validate-only",
                "--contracts-root",
                str(args.contracts_root),
            ]
        )
    for command in checks:
        _run_command(command, dry_run=False)
    _run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(REPO_ROOT / "tests"),
            "-v",
        ],
        dry_run=False,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.domains = list(dict.fromkeys(args.domains))
    args.providers = list(dict.fromkeys(args.providers))
    args.baselines = list(dict.fromkeys(args.baselines))
    args.contracts_root = args.contracts_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()

    if args.action == "validate":
        if args.phase is not None:
            parser.error("validate does not take a phase")
        _validate(args)
        print("Domain robustness validation passed.")
        return 0

    if args.phase is None:
        parser.error(f"{args.action} requires a phase: negotiation, performance, or all")
    commands = _commands(args, args.phase)
    if args.action == "plan":
        for command in commands:
            print(_render(command))
        return 0

    _validate_repo()
    _write_manifest(args, args.phase, commands)
    for command in commands:
        _run_command(command, dry_run=args.dry_run)
    print(f"Domain robustness {args.phase} run complete: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
