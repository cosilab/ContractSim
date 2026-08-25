"""Execute and persist exactly one LLM negotiation matchup."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

ORCHESTRATION_DIR = Path(__file__).resolve().parent
NEGOTIATION_DIR = ORCHESTRATION_DIR.parent
PROJECT_ROOT = NEGOTIATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_utils import load_environment_metadata  # noqa: E402
from llm_utils.artifacts import (  # noqa: E402
    RunClaim,
    claim_output_directory,
    write_json,
)
from llm_utils.llm import (  # noqa: E402
    LLMRuntimeBundle,
    PROVIDERS,
    RuntimeOptions,
    TraceRecorder,
    build_llm_runtime,
    save_agent_states,
)
from llm_utils.process_runner import (  # noqa: E402
    ProcessTermination,
    terminate_on_signals,
)
from negotiation.logic.negotiation import run_negotiation  # noqa: E402
from negotiation.prompting import (  # noqa: E402
    GOAL_TREATMENT_CHOICES,
    build_prompts,
    treatment_metadata,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True, choices=range(1, 7))
    parser.add_argument(
        "--customer-provider",
        required=True,
        choices=PROVIDERS,
    )
    parser.add_argument(
        "--caterer-provider",
        required=True,
        choices=PROVIDERS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=_positive_int, default=50)
    parser.add_argument(
        "--goal-treatment",
        choices=GOAL_TREATMENT_CHOICES,
        default="control",
    )
    parser.add_argument(
        "--scheduler-owned-output",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _environment_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _runtime_options(output_dir: Path) -> RuntimeOptions:
    return RuntimeOptions(
        prompt_cache_enabled=_environment_flag("PROMPT_CACHE", True),
        thought_traces_enabled=True,
        cache_scope=str(output_dir),
    )


def _validate_output_target(
    output_dir: Path,
    *,
    scheduler_owned: bool,
) -> None:
    """Reject stale direct-run artifacts while allowing the scheduler log."""
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise RuntimeError(f"output path is not a directory: {output_dir}")
    entries = list(output_dir.iterdir())
    allowed = {"stdout.log"} if scheduler_owned else set()
    unexpected = sorted(path.name for path in entries if path.name not in allowed)
    if unexpected:
        raise RuntimeError(
            f"output directory is not empty; choose a new run directory: "
            f"{output_dir}"
        )
    if scheduler_owned:
        log_path = output_dir / "stdout.log"
        try:
            log_stat = log_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"scheduler-owned output is missing stdout.log: {output_dir}"
            ) from exc
        if not stat.S_ISREG(log_stat.st_mode) or log_stat.st_mode & 0o077:
            raise RuntimeError(
                "scheduler-owned stdout.log must be a private regular file: "
                f"{log_path}"
            )


def _run_claimed(
    args: argparse.Namespace,
    *,
    environment: dict[str, Any],
    negotiation_players: list[dict[str, str]],
    goal_metadata: dict[str, object],
    runtime: LLMRuntimeBundle,
    options: RuntimeOptions,
    claim: RunClaim,
) -> int:
    output_dir = args.output_dir
    provider_configs = runtime.configs
    models = runtime.models
    agents = runtime.agents

    data_dir = output_dir / "data"
    config_dir = output_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / "negotiation_players.json", negotiation_players)

    trace_path = data_dir / "llm_reasoning_traces.jsonl"
    trace_recorder = TraceRecorder(trace_path, enabled=True)
    trace_recorder.reset()
    print(f"[reasoning trace log={trace_path}]")

    print(
        f"[environment={environment['environment_id']}({environment['environment_name']})  "
        f"customer={provider_configs[0].provider}({provider_configs[0].model})  "
        f"caterer={provider_configs[1].provider}({provider_configs[1].model})  "
        f"output={output_dir}]"
    )
    common_metadata = {
        **environment,
        "implementation": "Negotiation",
        "implementation_version": 5,
        "customer_provider": args.customer_provider,
        "customer_model": provider_configs[0].model,
        "caterer_provider": args.caterer_provider,
        "caterer_model": provider_configs[1].model,
        "max_rounds": args.max_rounds,
        "prompt_layout": "chat_v2",
        "prompt_template_version": "v1",
        "prompt_content_profile": (
            "original_goals_readable_delimited"
            if args.goal_treatment == "control"
            else "original_goals_plus_named_ablation_clause"
        ),
        **goal_metadata,
    }
    write_json(
        config_dir / "run_config.json",
        {
            **common_metadata,
            "customer_reasoning": provider_configs[0].reasoning_metadata,
            "caterer_reasoning": provider_configs[1].reasoning_metadata,
            "prompt_cache_enabled": options.prompt_cache_enabled,
            "thought_traces_enabled": True,
            "prompt_cache_providers": list(PROVIDERS),
            "prompt_cache_modes": {
                "gemini": "provider_implicit",
                "openai": "implicit",
                "anthropic": "automatic",
            },
            "empty_response_policy": "retry_once_then_invalid",
            "output_dir": str(output_dir),
        },
    )

    contract, accepted, run_status = run_negotiation(
        agents=agents,
        models=models,
        log_path=str(data_dir / "negotiation_log.txt"),
        contract_path=str(data_dir / "contract.txt"),
        max_rounds=args.max_rounds,
        record_llm_call=trace_recorder.record,
    )
    save_agent_states(
        output_dir,
        agents,
        provider_configs,
    )
    summary = {
        **common_metadata,
        **run_status,
        "accepted": accepted,
        "contract_text": contract or "",
    }
    write_json(data_dir / "summary.json", summary)
    terminal_status = "complete" if run_status["valid"] else "invalid"
    claim.finalize(
        terminal_status,
        {
            **run_status,
            "accepted": accepted,
            "environment_id": environment["environment_id"],
        },
    )
    return 0


def run_worker(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    _validate_output_target(
        output_dir,
        scheduler_owned=args.scheduler_owned_output,
    )
    args.output_dir = output_dir

    environment = load_environment_metadata(args.env)
    negotiation_players = build_prompts(
        args.env,
        goal_treatment=args.goal_treatment,
        max_rounds=args.max_rounds,
    )
    goal_metadata = treatment_metadata(args.goal_treatment)
    options = _runtime_options(output_dir)
    runtime = build_llm_runtime(
        negotiation_players,
        (args.customer_provider, args.caterer_provider),
        options=options,
    )

    claim = claim_output_directory(
        output_dir,
        metadata={
            "implementation": "Negotiation",
            "environment_id": environment["environment_id"],
        },
    )
    with claim:
        return _run_claimed(
            args,
            environment=environment,
            negotiation_players=negotiation_players,
            goal_metadata=goal_metadata,
            runtime=runtime,
            options=options,
            claim=claim,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        with terminate_on_signals():
            return run_worker(args)
    except ProcessTermination as exc:
        return exc.exit_code
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
