from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from negotiation.orchestration import run as public_run  # noqa: E402
from negotiation.orchestration import batch as batch_module, worker as worker_module  # noqa: E402
from llm_utils.llm import ProviderConfig  # noqa: E402


class PublicCliTests(unittest.TestCase):
    def test_single_mode_accepts_one_pairing_across_environments(self) -> None:
        args = public_run.build_parser().parse_args(
            [
                "single",
                "--envs",
                "1",
                "5",
                "--match",
                "openai",
                "anthropic",
            ]
        )
        self.assertEqual(args.mode, "single")
        self.assertEqual(args.envs, [1, 5])
        self.assertEqual(args.match, ["openai", "anthropic"])

    def test_batch_mode_separates_replicates_from_concurrency(self) -> None:
        args = public_run.build_parser().parse_args(
            [
                "batch",
                "--envs",
                "1",
                "4",
                "--runs",
                "3",
                "--max-concurrency",
                "7",
            ]
        )
        self.assertEqual(args.runs, 3)
        self.assertEqual(args.max_concurrency, 7)

    def test_public_entrypoint_prepares_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            argv = [
                "run.py",
                "single",
                "--envs",
                "1",
                "--match",
                "openai",
                "anthropic",
                "--results-root",
                str(results_root),
                "--max-rounds",
                "17",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    public_run,
                    "run_environments",
                    return_value=0,
                ) as environments,
            ):
                self.assertEqual(public_run.main(), 0)
            experiment_dir = environments.call_args.args[0].experiment_dir
            record = (experiment_dir / "run.txt").read_text()

            self.assertEqual(experiment_dir.parent.name, "negotiation-main")
            self.assertEqual(
                experiment_dir.parent.parent.resolve(),
                results_root.resolve(),
            )
            self.assertIn("started-at:", record)
            self.assertIn("command:", record)
            self.assertIn("--match openai anthropic", record)
            self.assertIn("goal-treatment: control", record)

            dry_root = Path(temp_dir) / "dry-results"
            dry_argv = list(argv)
            dry_argv[dry_argv.index("--results-root") + 1] = str(dry_root)
            dry_argv.append("--dry-run")
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    dry_argv,
                ),
                mock.patch.object(
                    public_run,
                    "run_environments",
                    return_value=0,
                ),
            ):
                self.assertEqual(public_run.main(), 0)
            self.assertFalse(dry_root.exists())

    def test_prompt_ablation_routes_to_negotiation_abl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir) / "results"
            argv = [
                "run.py",
                "single",
                "--envs",
                "2",
                "--match",
                "gemini",
                "openai",
                "--goal-treatment",
                "contingencies",
                "--results-root",
                str(results_root),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    public_run,
                    "run_environments",
                    return_value=0,
                ) as environments,
            ):
                self.assertEqual(public_run.main(), 0)

            experiment_dir = environments.call_args.args[0].experiment_dir
            record = (experiment_dir / "run.txt").read_text()
            self.assertEqual(experiment_dir.parent.name, "negotiation-abl")
            self.assertIn("goal-treatment: contingencies", record)


class EnvironmentSequencingTests(unittest.TestCase):
    def test_environments_run_in_series_in_requested_order(self) -> None:
        args = argparse.Namespace(
            mode="batch",
            envs=[5, 1, 4],
            providers=["gemini", "openai"],
            runs=2,
            max_concurrency=3,
            max_rounds=10,
            goal_treatment="control",
            experiment_dir=Path("/tmp/negotiation-tests/one"),
            dry_run=False,
        )
        with mock.patch.object(public_run, "run_batch", return_value=0) as batch:
            result = public_run.run_environments(args)

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.kwargs["environment_id"] for call in batch.call_args_list],
            [5, 1, 4],
        )
        self.assertTrue(
            all(call.kwargs["runs"] == 2 for call in batch.call_args_list)
        )

    def test_failure_stops_later_environments(self) -> None:
        args = argparse.Namespace(
            mode="single",
            envs=[1, 4, 5],
            match=["openai", "anthropic"],
            max_rounds=10,
            goal_treatment="control",
            experiment_dir=Path("/tmp/negotiation-tests/two"),
            dry_run=False,
        )
        with mock.patch.object(
            public_run, "run_batch", side_effect=[0, 1]
        ) as batch:
            result = public_run.run_environments(args)

        self.assertEqual(result, 1)
        self.assertEqual(batch.call_count, 2)
        self.assertEqual(batch.call_args_list[0].kwargs["max_concurrency"], 1)
        self.assertEqual(batch.call_args_list[0].kwargs["runs"], 1)


class BatchPlanningTests(unittest.TestCase):
    def test_runs_expand_pairings_without_prefixed_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "experiment"
            jobs = batch_module.build_jobs(
                environment_id=5,
                pairings=[("gemini", "openai"), ("openai", "gemini")],
                runs=3,
                environment_dir=experiment_dir,
                max_rounds=20,
                goal_treatment="control",
            )

        self.assertEqual(len(jobs), 6)
        self.assertEqual([job.run_number for job in jobs], [1, 1, 2, 2, 3, 3])
        command = jobs[0].command
        self.assertIn("--customer-provider", command)
        self.assertIn("--caterer-provider", command)
        self.assertIn("--scheduler-owned-output", command)
        forbidden_prefix = "CA" + "IRF_"
        self.assertFalse(any(forbidden_prefix in part for part in command))

    def test_jobs_are_projected_to_shared_runner_and_classified(self) -> None:
        captured: dict[str, object] = {}

        def fake_runner(process_jobs, **kwargs):
            specs = list(process_jobs)
            captured["specs"] = specs
            captured["max_concurrency"] = kwargs["max_concurrency"]
            results = []
            for index, spec in enumerate(specs, start=1):
                summary = spec.output_dir / "data" / "summary.json"
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text(
                    json.dumps(
                        {
                            "valid": True,
                            "accepted": False,
                            "turns_taken": 1,
                        }
                    )
                )
                kwargs["on_start"](spec, 1000 + index)
                result = batch_module.ProcessResult(spec, 1000 + index, 0, 0.1)
                results.append(result)
                kwargs["on_complete"](result)
            return results

        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = batch_module.build_jobs(
                environment_id=1,
                pairings=[("gemini", "gemini"), ("openai", "openai")],
                runs=1,
                environment_dir=Path(temp_dir),
                max_rounds=5,
                goal_treatment="control",
            )
            with mock.patch.object(
                batch_module,
                "run_process_jobs",
                side_effect=fake_runner,
            ):
                failed, invalid = batch_module._run_jobs(jobs, max_concurrency=2)

        self.assertEqual(failed, [])
        self.assertEqual(invalid, [])
        self.assertEqual(captured["max_concurrency"], 2)
        specs = captured["specs"]
        self.assertEqual([spec.label for spec in specs], [job.label for job in jobs])
        self.assertTrue(all(spec.log_path.name == "stdout.log" for spec in specs))

    def test_dry_run_does_not_create_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "results" / "timestamp"
            result = batch_module.run_batch(
                environment_id=1,
                pairings=[("gemini", "openai")],
                runs=2,
                max_rounds=5,
                goal_treatment="control",
                experiment_dir=experiment_dir,
                max_concurrency=1,
                dry_run=True,
            )
            self.assertEqual(result, 0)
            self.assertFalse(experiment_dir.exists())

    def test_direct_batch_api_resolves_relative_experiment_directory(self) -> None:
        relative_dir = Path("relative-negotiation-results")
        with mock.patch.object(
            batch_module,
            "build_jobs",
            wraps=batch_module.build_jobs,
        ) as build_jobs:
            result = batch_module.run_batch(
                environment_id=1,
                pairings=[("gemini", "openai")],
                runs=1,
                max_rounds=5,
                goal_treatment="control",
                experiment_dir=relative_dir,
                max_concurrency=1,
                dry_run=True,
            )

        self.assertEqual(result, 0)
        environment_dir = build_jobs.call_args.kwargs["environment_dir"]
        self.assertTrue(environment_dir.is_absolute())
        self.assertEqual(
            environment_dir,
            relative_dir.resolve() / "env-1",
        )


class BatchLifecycleTests(unittest.TestCase):
    def test_process_termination_maps_to_shell_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                batch_module,
                "_run_jobs",
                side_effect=batch_module.ProcessTermination(signal.SIGTERM),
            ):
                result = batch_module.run_batch(
                    environment_id=1,
                    pairings=[("gemini", "openai")],
                    runs=1,
                    max_rounds=5,
                    goal_treatment="control",
                    experiment_dir=Path(temp_dir),
                    max_concurrency=1,
                )

        self.assertEqual(result, 128 + signal.SIGTERM)

    def test_keyboard_interrupt_still_returns_130(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                batch_module, "_run_jobs", side_effect=KeyboardInterrupt
            ):
                result = batch_module.run_batch(
                    environment_id=1,
                    pairings=[("gemini", "openai")],
                    runs=1,
                    max_rounds=5,
                    goal_treatment="control",
                    experiment_dir=Path(temp_dir),
                    max_concurrency=1,
                )

        self.assertEqual(result, 130)


class BatchSummaryValidationTests(unittest.TestCase):
    def _status_for(self, summary: object) -> tuple[str, str | None]:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary_path = output_dir / "data" / "summary.json"
            summary_path.parent.mkdir()
            summary_path.write_text(json.dumps(summary))
            job = batch_module.NegotiationJob(
                run_number=1,
                customer_provider="gemini",
                caterer_provider="openai",
                output_dir=output_dir,
                command=("worker",),
            )
            return batch_module._job_status(job, 0)

    def test_valid_and_explicitly_invalid_summaries_are_classified(self) -> None:
        self.assertEqual(
            self._status_for(
                {
                    "valid": True,
                    "accepted": False,
                    "turns_taken": 0,
                    "contract_text": "",
                }
            ),
            ("OK", None),
        )
        self.assertEqual(
            self._status_for(
                {
                    "valid": False,
                    "accepted": False,
                    "turns_taken": 2,
                    "invalid_reason": "empty_model_response",
                }
            ),
            ("INVALID (empty_model_response)", "invalid"),
        )

    def test_accepted_summary_requires_nonempty_contract_text(self) -> None:
        self.assertEqual(
            self._status_for(
                {
                    "valid": True,
                    "accepted": True,
                    "turns_taken": 2,
                    "contract_text": "Pizza: $10",
                }
            ),
            ("OK", None),
        )
        for contract_text in (None, "", "   ", 10):
            with self.subTest(contract_text=contract_text):
                status, category = self._status_for(
                    {
                        "valid": True,
                        "accepted": True,
                        "turns_taken": 2,
                        "contract_text": contract_text,
                    }
                )
                self.assertEqual(
                    status,
                    "FAILED (invalid summary: accepted runs require nonempty "
                    "contract_text)",
                )
                self.assertEqual(category, "failed")

    def test_malformed_summary_schema_is_failed(self) -> None:
        malformed_summaries = [
            [],
            {},
            {"valid": 1, "accepted": False, "turns_taken": 0},
            {"valid": True, "accepted": "false", "turns_taken": 0},
            {"valid": True, "accepted": False, "turns_taken": -1},
            {
                "valid": False,
                "accepted": False,
                "turns_taken": 0,
                "invalid_reason": " ",
            },
        ]
        for summary in malformed_summaries:
            with self.subTest(summary=summary):
                status, category = self._status_for(summary)
                self.assertTrue(status.startswith("FAILED (invalid summary:"))
                self.assertEqual(category, "failed")

    def test_invalid_utf8_summary_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary_path = output_dir / "data" / "summary.json"
            summary_path.parent.mkdir()
            summary_path.write_bytes(b"\xff\xfe\x00")
            job = batch_module.NegotiationJob(
                run_number=1,
                customer_provider="gemini",
                caterer_provider="openai",
                output_dir=output_dir,
                command=("worker",),
            )

            status, category = batch_module._job_status(job, 0)

        self.assertTrue(status.startswith("FAILED (invalid summary:"))
        self.assertEqual(category, "failed")


class WorkerConfigurationTests(unittest.TestCase):
    def test_worker_always_enables_reasoning_traces(self) -> None:
        args = worker_module.build_parser().parse_args(
            [
                "--env",
                "5",
                "--customer-provider",
                "openai",
                "--caterer-provider",
                "anthropic",
                "--output-dir",
                "/tmp/negotiation-worker",
            ]
        )
        options = worker_module._runtime_options(args.output_dir)
        self.assertTrue(options.thought_traces_enabled)
        self.assertEqual(options.cache_scope, "/tmp/negotiation-worker")

    def test_worker_wires_prompts_models_agents_and_artifacts(self) -> None:
        class FakeAgent:
            def __init__(self, name: str) -> None:
                self.name = name

        fake_agents = [FakeAgent("customer"), FakeAgent("caterer")]
        fake_models = [object(), object()]
        fake_configs = (
            ProviderConfig("openai", "gpt-test", "high"),
            ProviderConfig("anthropic", "claude-test", "high"),
        )
        runtime = SimpleNamespace(
            agents=tuple(fake_agents),
            models=tuple(fake_models),
            configs=fake_configs,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "worker"
            output_dir.mkdir()
            scheduler_log = output_dir / "stdout.log"
            scheduler_log.write_text("scheduler log\n", encoding="utf-8")
            scheduler_log.chmod(0o600)
            args = worker_module.build_parser().parse_args(
                [
                    "--env",
                    "2",
                    "--customer-provider",
                    "openai",
                    "--caterer-provider",
                    "anthropic",
                    "--output-dir",
                    str(output_dir),
                    "--max-rounds",
                    "13",
                    "--scheduler-owned-output",
                ]
            )
            with (
                mock.patch.object(
                    worker_module,
                    "build_llm_runtime",
                    return_value=runtime,
                ) as build_runtime,
                mock.patch.object(
                    worker_module,
                    "build_prompts",
                    wraps=worker_module.build_prompts,
                ) as build_prompts,
                mock.patch.object(
                    worker_module,
                    "run_negotiation",
                    return_value=(
                        "Pizza: $10",
                        True,
                        {"valid": True, "turns_taken": 4},
                    ),
                ) as negotiate,
                mock.patch.object(worker_module, "save_agent_states") as save_agents,
            ):
                result = worker_module.run_worker(args)

            run_config = json.loads(
                (output_dir / "config" / "run_config.json").read_text()
            )
            summary = json.loads(
                (output_dir / "data" / "summary.json").read_text()
            )
            status = json.loads((output_dir / "run_status.json").read_text())
            scheduler_log_contents = scheduler_log.read_text(encoding="utf-8")
            trace_path = output_dir / "data" / "llm_reasoning_traces.jsonl"
            trace_exists = trace_path.is_file()

        self.assertEqual(result, 0)
        self.assertEqual(
            build_runtime.call_args.args[1],
            ("openai", "anthropic"),
        )
        self.assertEqual(
            build_runtime.call_args.kwargs["options"].cache_scope,
            str(output_dir.resolve()),
        )
        build_prompts.assert_called_once_with(
            2,
            goal_treatment="control",
            max_rounds=13,
        )
        negotiate.assert_called_once()
        self.assertEqual(negotiate.call_args.kwargs["max_rounds"], 13)
        save_agents.assert_called_once()
        self.assertEqual(run_config["customer_provider"], "openai")
        self.assertEqual(run_config["caterer_provider"], "anthropic")
        self.assertEqual(run_config["max_rounds"], 13)
        self.assertTrue(run_config["thought_traces_enabled"])
        self.assertEqual(summary["max_rounds"], 13)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["contract_text"], "Pizza: $10")
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["valid"])
        self.assertTrue(trace_exists)
        self.assertEqual(scheduler_log_contents, "scheduler log\n")

    def test_worker_rejects_stale_direct_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "worker"
            output_dir.mkdir()
            marker = output_dir / "stdout.log"
            marker.write_text("stale log\n", encoding="utf-8")
            args = worker_module.build_parser().parse_args(
                [
                    "--env",
                    "1",
                    "--customer-provider",
                    "openai",
                    "--caterer-provider",
                    "anthropic",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                worker_module.run_worker(args)

            self.assertEqual(marker.read_text(encoding="utf-8"), "stale log\n")

    def test_scheduler_handoff_requires_a_private_regular_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "worker"
            output_dir.mkdir()
            log_path = output_dir / "stdout.log"
            log_path.write_text("public log\n", encoding="utf-8")
            log_path.chmod(0o644)
            args = worker_module.build_parser().parse_args(
                [
                    "--env",
                    "1",
                    "--customer-provider",
                    "openai",
                    "--caterer-provider",
                    "anthropic",
                    "--output-dir",
                    str(output_dir),
                    "--scheduler-owned-output",
                ]
            )

            with self.assertRaisesRegex(RuntimeError, "private regular file"):
                worker_module.run_worker(args)

    def test_post_claim_failure_records_error_status(self) -> None:
        fake_agents = (
            SimpleNamespace(name="customer"),
            SimpleNamespace(name="caterer"),
        )
        runtime = SimpleNamespace(
            agents=fake_agents,
            models=(object(), object()),
            configs=(
                ProviderConfig("openai", "gpt-test", "high"),
                ProviderConfig("anthropic", "claude-test", "high"),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "worker"
            args = worker_module.build_parser().parse_args(
                [
                    "--env",
                    "1",
                    "--customer-provider",
                    "openai",
                    "--caterer-provider",
                    "anthropic",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            with (
                mock.patch.object(
                    worker_module,
                    "build_llm_runtime",
                    return_value=runtime,
                ),
                mock.patch.object(
                    worker_module,
                    "run_negotiation",
                    side_effect=RuntimeError("post-claim failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "post-claim failure"),
            ):
                worker_module.run_worker(args)

            status = json.loads((output_dir / "run_status.json").read_text())
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
