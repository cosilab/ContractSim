from __future__ import annotations

import json
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from llm_utils import process_runner
from llm_utils.artifacts import PRIVATE_FILE_MODE, claim_output_directory
from llm_utils.process_runner import (
    ProcessJob,
    ProcessResult,
    ProcessTermination,
    run_process_jobs,
    terminate_on_signals,
)


class ProcessJobTests(unittest.TestCase):
    def test_specification_is_validated_normalized_and_immutable(self) -> None:
        arguments = ["worker", "--flag"]
        job = ProcessJob(
            label="job one",
            argv=arguments,  # type: ignore[arg-type]
            output_dir="output",  # type: ignore[arg-type]
            log_path="output/stdout.log",  # type: ignore[arg-type]
            cwd="work",  # type: ignore[arg-type]
        )
        arguments.append("--mutated")

        self.assertEqual(job.argv, ("worker", "--flag"))
        self.assertEqual(job.output_dir, Path("output"))
        self.assertEqual(job.log_path, Path("output/stdout.log"))
        self.assertEqual(job.cwd, Path("work"))
        with self.assertRaises(FrozenInstanceError):
            job.label = "changed"  # type: ignore[misc]

        invalid_specs = [
            {"label": "", "argv": ("worker",)},
            {"label": "job", "argv": ()},
            {"label": "job", "argv": ("worker", 2)},
        ]
        for overrides in invalid_specs:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    ProcessJob(
                        output_dir=Path("output"),
                        log_path=Path("output/stdout.log"),
                        **overrides,  # type: ignore[arg-type]
                    )


class ProcessExecutionTests(unittest.TestCase):
    def test_real_process_captures_combined_output_and_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            log_path = output_dir / "logs" / "worker.log"
            job = ProcessJob(
                label="real worker",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import os, sys; print(os.getcwd()); "
                        "print('stderr line', file=sys.stderr)"
                    ),
                ),
                output_dir=output_dir,
                log_path=log_path,
                cwd=root,
            )
            starts: list[tuple[ProcessJob, int]] = []
            completions: list[ProcessResult] = []

            results = run_process_jobs(
                [job],
                max_concurrency=1,
                on_start=lambda started_job, pid: starts.append(
                    (started_job, pid)
                ),
                on_complete=completions.append,
            )

            log_text = log_path.read_text(encoding="utf-8")
            log_mode = stat.S_IMODE(log_path.stat().st_mode)

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].job, job)
        self.assertTrue(results[0].succeeded)
        self.assertGreater(results[0].pid, 0)
        self.assertGreaterEqual(results[0].elapsed_seconds, 0.0)
        self.assertEqual(starts, [(job, results[0].pid)])
        self.assertEqual(completions, results)
        self.assertIn(str(root), log_text)
        self.assertIn("stderr line", log_text)
        self.assertEqual(log_mode, PRIVATE_FILE_MODE)

    def test_scheduler_bounds_concurrency_and_preserves_input_order(self) -> None:
        active = 0
        maximum_active = 0
        popen_calls: list[tuple[list[str], dict[str, object]]] = []

        class FakeProcess:
            next_pid = 1000

            def __init__(self, argv: list[str]) -> None:
                nonlocal active, maximum_active
                FakeProcess.next_pid += 1
                self.pid = FakeProcess.next_pid
                self.returncode: int | None = None
                self.expected_returncode = int(argv[-1])
                active += 1
                maximum_active = max(maximum_active, active)

            def poll(self) -> int | None:
                nonlocal active
                if self.returncode is None:
                    self.returncode = self.expected_returncode
                    active -= 1
                return self.returncode

        def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
            popen_calls.append((argv, kwargs))
            return FakeProcess(argv)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = [
                ProcessJob(
                    label=f"job {index}",
                    argv=("worker", str(returncode)),
                    output_dir=root / f"output-{index}",
                    log_path=root / f"output-{index}" / "stdout.log",
                    cwd=root,
                )
                for index, returncode in enumerate((0, 7, 0))
            ]
            started: list[str] = []
            completed: list[str] = []
            with mock.patch.object(
                process_runner.subprocess,
                "Popen",
                side_effect=fake_popen,
            ):
                results = run_process_jobs(
                    jobs,
                    max_concurrency=2,
                    on_start=lambda job, _pid: started.append(job.label),
                    on_complete=lambda result: completed.append(
                        result.job.label
                    ),
                )

        self.assertEqual(maximum_active, 2)
        self.assertEqual([result.job for result in results], jobs)
        self.assertEqual(
            [result.returncode for result in results],
            [0, 7, 0],
        )
        self.assertEqual(started, ["job 0", "job 1", "job 2"])
        self.assertEqual(completed, ["job 0", "job 1", "job 2"])
        for _argv, kwargs in popen_calls:
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(kwargs["cwd"], root)

    def test_existing_output_directory_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "existing"
            output_dir.mkdir()
            marker = output_dir / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            job = ProcessJob(
                label="collision",
                argv=(sys.executable, "-c", "pass"),
                output_dir=output_dir,
                log_path=output_dir / "stdout.log",
            )

            with self.assertRaises(FileExistsError):
                run_process_jobs([job], max_concurrency=1)

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertFalse(job.log_path.exists())

    def test_argument_validation_and_empty_input(self) -> None:
        for concurrency in (True, False, 0, -1, 1.5, "1", None):
            with self.subTest(concurrency=concurrency):
                with self.assertRaises(ValueError):
                    run_process_jobs(
                        [],
                        max_concurrency=concurrency,  # type: ignore[arg-type]
                    )
        for grace in (True, False, -0.1, float("nan"), float("inf"), "1", None):
            with self.subTest(grace=grace):
                with self.assertRaises(ValueError):
                    run_process_jobs(
                        [],
                        max_concurrency=1,
                        shutdown_grace_seconds=grace,  # type: ignore[arg-type]
                    )

        self.assertEqual(run_process_jobs([], max_concurrency=1), [])
        with self.assertRaises(TypeError):
            run_process_jobs(
                [mock.sentinel.not_a_job],  # type: ignore[list-item]
                max_concurrency=1,
            )

    def test_leaf_worker_signal_guard_raises_and_restores_handlers(self) -> None:
        installed: dict[int, object] = {}
        previous = object()

        def install(signum: int, handler: object) -> object:
            installed[signum] = handler
            return previous

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "claimed-worker"
            with (
                mock.patch.object(signal, "getsignal", return_value=previous),
                mock.patch.object(
                    signal,
                    "signal",
                    side_effect=install,
                ) as set_signal,
                self.assertRaises(ProcessTermination) as caught,
            ):
                with claim_output_directory(output_dir):
                    with terminate_on_signals():
                        handler = installed[signal.SIGTERM]
                        assert callable(handler)
                        handler(signal.SIGTERM, None)

            status = json.loads(
                (output_dir / "run_status.json").read_text(encoding="utf-8")
            )

        self.assertEqual(caught.exception.signum, signal.SIGTERM)
        self.assertEqual(status["status"], "error")
        self.assertEqual(status["error"]["type"], "ProcessTermination")
        self.assertIn(
            mock.call(signal.SIGTERM, previous),
            set_signal.call_args_list,
        )


@unittest.skipUnless(
    hasattr(signal, "SIGHUP") and hasattr(process_runner.os, "killpg"),
    "process-group signal tests require POSIX",
)
class ProcessSignalTests(unittest.TestCase):
    class SignalProcess:
        next_pid = 4100

        def __init__(self) -> None:
            type(self).next_pid += 1
            self.pid = type(self).next_pid
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

    def test_sigterm_and_sighup_are_recorded_cleaned_and_restored(self) -> None:
        for requested_signal in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(requested_signal=requested_signal):
                process = self.SignalProcess()
                group_exists = True
                previous_handler = signal.getsignal(requested_signal)

                def launch_and_signal(
                    _argv: list[str], **_kwargs: object
                ) -> ProcessSignalTests.SignalProcess:
                    handler = signal.getsignal(requested_signal)
                    self.assertTrue(callable(handler))
                    handler(requested_signal, None)
                    return process

                def fake_killpg(process_group: int, signum: int) -> None:
                    nonlocal group_exists
                    self.assertEqual(process_group, process.pid)
                    if signum == 0:
                        if not group_exists:
                            raise ProcessLookupError
                        return
                    process.returncode = -signum
                    group_exists = False

                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir) / "output"
                    job = ProcessJob(
                        label="signal worker",
                        argv=("worker",),
                        output_dir=output_dir,
                        log_path=output_dir / "stdout.log",
                    )
                    with (
                        mock.patch.object(
                            process_runner.subprocess,
                            "Popen",
                            side_effect=launch_and_signal,
                        ) as popen,
                        mock.patch.object(
                            process_runner.os,
                            "killpg",
                            side_effect=fake_killpg,
                        ) as killpg,
                        self.assertRaises(ProcessTermination) as caught,
                    ):
                        run_process_jobs(
                            [job],
                            max_concurrency=1,
                            shutdown_grace_seconds=0.0,
                        )

                self.assertEqual(caught.exception.signum, requested_signal)
                self.assertEqual(
                    caught.exception.exit_code,
                    128 + requested_signal,
                )
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(
                    killpg.call_args_list,
                    [
                        mock.call(process.pid, signal.SIGTERM),
                        mock.call(process.pid, 0),
                    ],
                )
                self.assertIs(
                    signal.getsignal(requested_signal), previous_handler
                )

    def test_keyboard_interrupt_from_callback_stops_registered_worker(self) -> None:
        process = self.SignalProcess()
        group_exists = True

        def fake_killpg(process_group: int, signum: int) -> None:
            nonlocal group_exists
            self.assertEqual(process_group, process.pid)
            if signum == 0:
                if not group_exists:
                    raise ProcessLookupError
                return
            process.returncode = -signum
            group_exists = False

        def interrupt_after_start(_job: ProcessJob, _pid: int) -> None:
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            job = ProcessJob(
                label="interrupt worker",
                argv=("worker",),
                output_dir=output_dir,
                log_path=output_dir / "stdout.log",
            )
            with (
                mock.patch.object(
                    process_runner.subprocess,
                    "Popen",
                    return_value=process,
                ),
                mock.patch.object(
                    process_runner.os,
                    "killpg",
                    side_effect=fake_killpg,
                ) as killpg,
                self.assertRaises(KeyboardInterrupt),
            ):
                run_process_jobs(
                    [job],
                    max_concurrency=1,
                    shutdown_grace_seconds=0.0,
                    on_start=interrupt_after_start,
                )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(process.pid, signal.SIGTERM),
                mock.call(process.pid, 0),
            ],
        )

    def test_grace_expiry_force_kills_surviving_process_group(self) -> None:
        process = self.SignalProcess()
        process.returncode = 0  # The leader exited, but a descendant remains.
        group_exists = True

        def fake_killpg(process_group: int, signum: int) -> None:
            nonlocal group_exists
            self.assertEqual(process_group, process.pid)
            if signum == 0:
                if not group_exists:
                    raise ProcessLookupError
            elif signum == signal.SIGKILL:
                group_exists = False

        with mock.patch.object(
            process_runner.os,
            "killpg",
            side_effect=fake_killpg,
        ) as killpg:
            process_runner._stop_processes(
                {process: mock.sentinel.running},  # type: ignore[dict-item]
                shutdown_grace_seconds=0.0,
            )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(process.pid, signal.SIGTERM),
                mock.call(process.pid, 0),
                mock.call(process.pid, signal.SIGKILL),
            ],
        )

    def test_completion_callback_failure_cleans_other_workers(self) -> None:
        first = self.SignalProcess()
        second = self.SignalProcess()
        first.returncode = 0
        group_exists = {first.pid: False, second.pid: True}

        def choose_once(
            _argv: list[str], **_kwargs: object
        ) -> ProcessSignalTests.SignalProcess:
            if not hasattr(choose_once, "called"):
                choose_once.called = True  # type: ignore[attr-defined]
                return first
            return second

        def fake_killpg(process_group: int, signum: int) -> None:
            if signum == 0:
                if not group_exists[process_group]:
                    raise ProcessLookupError
                return
            group_exists[process_group] = False
            if process_group == second.pid:
                second.returncode = -signum

        def fail_reporting(_result: ProcessResult) -> None:
            raise RuntimeError("reporting failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = [
                ProcessJob(
                    label=f"job {index}",
                    argv=("worker",),
                    output_dir=root / f"output-{index}",
                    log_path=root / f"output-{index}" / "stdout.log",
                )
                for index in range(2)
            ]
            with (
                mock.patch.object(
                    process_runner.subprocess,
                    "Popen",
                    side_effect=choose_once,
                ),
                mock.patch.object(
                    process_runner.os,
                    "killpg",
                    side_effect=fake_killpg,
                ) as killpg,
                self.assertRaisesRegex(RuntimeError, "reporting failed"),
            ):
                run_process_jobs(
                    jobs,
                    max_concurrency=2,
                    shutdown_grace_seconds=0.0,
                    on_complete=fail_reporting,
                )

        self.assertIn(
            mock.call(second.pid, signal.SIGTERM),
            killpg.call_args_list,
        )


class ProcessFallbackCleanupTests(unittest.TestCase):
    def test_non_posix_cleanup_uses_direct_process_methods(self) -> None:
        process = mock.Mock(pid=5100)
        process.poll.return_value = None
        process.wait.return_value = 0

        with (
            mock.patch.object(process_runner.os, "name", "nt"),
            mock.patch.object(process_runner.os, "killpg") as killpg,
        ):
            process_runner._stop_processes(
                {process: mock.sentinel.running},  # type: ignore[dict-item]
                shutdown_grace_seconds=0.0,
            )

        killpg.assert_not_called()
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
