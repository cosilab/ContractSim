from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from performance.orchestration import batch as batch_runner
from performance.orchestration import run as public_run


class EnvironmentSelectionTests(unittest.TestCase):
    def test_batch_accepts_one_or_multiple_environments(self) -> None:
        parser = public_run.build_parser()
        shared = [
            "--customers",
            "rc",
            "--suppliers",
            "openai",
        ]
        one = parser.parse_args(["batch", "--env", "3", *shared])
        many = parser.parse_args(
            ["batch", "--envs", "5", "1", "4", *shared]
        )
        resume = parser.parse_args(
            ["batch", "--resume-run-dir", "/tmp/campaign", *shared]
        )

        self.assertEqual(public_run._requested_environments(one), [3])
        self.assertEqual(public_run._requested_environments(many), [5, 1, 4])
        self.assertEqual(resume.resume_run_dir, Path("/tmp/campaign"))


class EnvironmentSequencingTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            mode="batch",
            customers=["rc", "rcc"],
            suppliers=["openai"],
            seeds=[42],
            max_concurrency=60,
            goal_treatment="control",
            dry_run=False,
            resume_run_dir=None,
        )

    def test_environments_run_in_requested_order(self) -> None:
        environments = [5, 1, 4]
        contracts = {
            environment_id: [Path(f"contract-{environment_id}")]
            for environment_id in environments
        }
        experiment = Path("/tmp/performance-sequencing")

        with mock.patch.object(public_run, "run_batch", return_value=0) as batch:
            result = public_run.run_environments(
                self._args(), environments, contracts, experiment
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.kwargs["environment_id"] for call in batch.call_args_list],
            environments,
        )
        self.assertEqual(
            [call.kwargs["experiment_dir"] for call in batch.call_args_list],
            [experiment / f"env-{item}" for item in environments],
        )
        self.assertTrue(
            all(
                call.kwargs["max_concurrency"] == 60
                for call in batch.call_args_list
            )
        )

    def test_failure_stops_later_environments(self) -> None:
        environments = [1, 2, 3]
        contracts = {
            environment_id: [Path(f"contract-{environment_id}")]
            for environment_id in environments
        }

        with mock.patch.object(
            public_run,
            "run_batch",
            side_effect=[0, 1],
        ) as batch:
            result = public_run.run_environments(
                self._args(),
                environments,
                contracts,
                Path("/tmp/performance-failure"),
            )

        self.assertEqual(result, 1)
        self.assertEqual(batch.call_count, 2)


class ResumeTests(unittest.TestCase):
    def test_resume_skips_complete_job_and_archives_incomplete_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "campaign" / "env-1"
            contract = root / "contract-1"
            complete = (
                experiment
                / contract.name
                / "customer-rc_supplier-gemini_seed-42"
            )
            incomplete = (
                experiment
                / contract.name
                / "customer-rcc_supplier-gemini_seed-42"
            )
            for output in (complete, incomplete):
                output.mkdir(parents=True)
            (complete / "run_status.json").write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )
            (complete / "data").mkdir()
            (complete / "data" / "summary.json").write_text(
                json.dumps({"valid": True}),
                encoding="utf-8",
            )
            (incomplete / "run_status.json").write_text(
                json.dumps({"status": "error"}),
                encoding="utf-8",
            )
            stale = root / "stale" / "env-1"

            with mock.patch.object(batch_runner, "run_process_jobs") as runner:
                result = batch_runner.run_batch(
                    contract_dirs=[contract],
                    customers=["rc", "rcc"],
                    suppliers=["gemini"],
                    seeds=[42],
                    environment_id=1,
                    goal_treatment="control",
                    experiment_dir=experiment,
                    max_concurrency=1,
                    resume=True,
                    resume_stale_dir=stale,
                )

            self.assertEqual(result, 0)
            self.assertTrue(complete.is_dir())
            self.assertFalse(incomplete.exists())
            self.assertTrue(
                (
                    stale / contract.name / incomplete.name / "run_status.json"
                ).is_file()
            )
            jobs = list(runner.call_args.args[0])
            self.assertEqual(len(jobs), 1)
            self.assertIn("customer-rcc_supplier-gemini", jobs[0].label)


if __name__ == "__main__":
    unittest.main()
