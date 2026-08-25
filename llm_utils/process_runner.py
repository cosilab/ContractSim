"""Bounded subprocess execution with orderly process-group cleanup.

This module owns only generic process lifecycle concerns.  Callers define jobs,
interpret return codes and artifacts, and provide any user-facing reporting.
"""

from __future__ import annotations

import math
import numbers
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .artifacts import PRIVATE_FILE_MODE


DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.1

__all__ = (
    "DEFAULT_SHUTDOWN_GRACE_SECONDS",
    "ProcessCompleteCallback",
    "ProcessJob",
    "ProcessResult",
    "ProcessStartCallback",
    "ProcessTermination",
    "run_process_jobs",
    "terminate_on_signals",
)


@dataclass(frozen=True, slots=True)
class ProcessJob:
    """Immutable specification for one subprocess invocation."""

    label: str
    argv: tuple[str, ...]
    output_dir: Path
    log_path: Path
    cwd: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a nonempty string")

        argv = tuple(self.argv)
        if not argv:
            raise ValueError("argv must contain at least one argument")
        if not all(isinstance(argument, str) for argument in argv):
            raise TypeError("argv arguments must all be strings")

        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "log_path", Path(self.log_path))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Completed process metadata, independent of any artifact schema."""

    job: ProcessJob
    pid: int
    returncode: int
    elapsed_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class ProcessTermination(BaseException):
    """Raised after SIGTERM or SIGHUP is recorded and workers are stopped."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum

    @property
    def exit_code(self) -> int:
        return 128 + self.signum


ProcessStartCallback = Callable[[ProcessJob, int], None]
ProcessCompleteCallback = Callable[[ProcessResult], None]


@contextmanager
def terminate_on_signals() -> Iterator[None]:
    """Raise :class:`ProcessTermination` immediately on SIGTERM or SIGHUP.

    This guard is intended for a leaf worker with no child-registration race.
    A run claim surrounding the worker body can then persist the interruption
    before the process exits. The batch scheduler uses its separate deferred
    signal guard instead.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers: dict[int, signal.Handlers] = {}

    def raise_termination(signum: int, _frame: object) -> None:
        raise ProcessTermination(signum)

    try:
        for signum in handled_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, raise_termination)
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


@dataclass
class _PendingTermination:
    signum: int | None = None

    def request(self, signum: int) -> None:
        if self.signum is None:
            self.signum = signum

    def raise_if_requested(self) -> None:
        if self.signum is not None:
            raise ProcessTermination(self.signum)


@contextmanager
def _termination_signal_handlers() -> Iterator[_PendingTermination]:
    """Record termination signals so launch registration remains atomic."""

    pending = _PendingTermination()
    if threading.current_thread() is not threading.main_thread():
        yield pending
        return

    handled_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers: dict[int, signal.Handlers] = {}

    def record_termination(signum: int, _frame: object) -> None:
        # Raising from a handler can strand a child if delivery lands between
        # Popen creating it and the scheduler registering it.  Record first,
        # then raise at an explicit safe point.
        pending.request(signum)

    try:
        for signum in handled_signals:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, record_termination)
        yield pending
        pending.raise_if_requested()
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


@dataclass(slots=True)
class _RunningProcess:
    index: int
    job: ProcessJob
    log_file: TextIO
    started_at: float


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def _open_private_log(path: Path) -> TextIO:
    """Exclusively create one scheduler log with private permissions."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _signal_process_group(
    process: subprocess.Popen[bytes],
    *,
    force: bool,
) -> None:
    """Signal an isolated process group, with a direct-process fallback."""

    group_error: OSError | None = None
    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            signum = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except OSError as exc:
            group_error = exc

    try:
        if force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except OSError as direct_error:
        group_detail = (
            f"process group: {group_error}; "
            if group_error is not None
            else ""
        )
        _warn(
            f"unable to signal process PID {process.pid}: "
            f"{group_detail}process: {direct_error}"
        )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        _warn(f"unable to inspect process group {process_group}: {exc}")
        return True
    return True


def _reap_process_leaders(
    processes: list[subprocess.Popen[bytes]],
    *,
    timeout: float,
) -> None:
    """Reap tracked leaders without extending one shared deadline."""

    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _warn(f"process PID {process.pid} did not exit after forced cleanup")


def _stop_processes(
    running: dict[subprocess.Popen[bytes], _RunningProcess],
    *,
    shutdown_grace_seconds: float,
) -> None:
    """Terminate every tracked worker, then force-kill grace-period survivors."""

    tracked_processes = list(running)
    if os.name == "posix" and hasattr(os, "killpg"):
        # PIDs are also process-group IDs because every child starts a new
        # session.  Signal groups even when a leader has already exited, since
        # descendants can keep its group alive.
        for process in tracked_processes:
            _signal_process_group(process, force=False)

        remaining_groups = [process.pid for process in tracked_processes]
        deadline = time.monotonic() + shutdown_grace_seconds
        while remaining_groups:
            for process in tracked_processes:
                process.poll()
            remaining_groups = [
                process_group
                for process_group in remaining_groups
                if _process_group_exists(process_group)
            ]
            if not remaining_groups:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

        tracked_by_pid = {
            process.pid: process for process in tracked_processes
        }
        for process_group in remaining_groups:
            _signal_process_group(tracked_by_pid[process_group], force=True)
        _reap_process_leaders(
            tracked_processes,
            timeout=shutdown_grace_seconds,
        )
        return

    live_processes = [
        process for process in tracked_processes if process.poll() is None
    ]
    for process in live_processes:
        _signal_process_group(process, force=False)

    deadline = time.monotonic() + shutdown_grace_seconds
    survivors: list[subprocess.Popen[bytes]] = []
    for process in live_processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            survivors.append(process)

    for process in survivors:
        _signal_process_group(process, force=True)
    _reap_process_leaders(survivors, timeout=shutdown_grace_seconds)


def _run_scheduler(
    jobs: list[ProcessJob],
    *,
    max_concurrency: int,
    shutdown_grace_seconds: float,
    termination: _PendingTermination,
    on_start: ProcessStartCallback | None,
    on_complete: ProcessCompleteCallback | None,
) -> list[ProcessResult]:
    pending = deque(enumerate(jobs))
    running: dict[subprocess.Popen[bytes], _RunningProcess] = {}
    results: list[ProcessResult | None] = [None] * len(jobs)

    try:
        while pending or running:
            termination.raise_if_requested()
            while pending and len(running) < max_concurrency:
                termination.raise_if_requested()
                index, job = pending.popleft()
                job.output_dir.mkdir(parents=True, exist_ok=False)
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = _open_private_log(job.log_path)
                process: subprocess.Popen[bytes] | None = None
                try:
                    termination.raise_if_requested()
                    process = subprocess.Popen(
                        list(job.argv),
                        cwd=job.cwd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    running[process] = _RunningProcess(
                        index=index,
                        job=job,
                        log_file=log_file,
                        started_at=time.monotonic(),
                    )
                    termination.raise_if_requested()
                    if on_start is not None:
                        on_start(job, process.pid)
                    termination.raise_if_requested()
                except BaseException:
                    if process is None:
                        log_file.close()
                    else:
                        # Keep a child discoverable if interruption or a
                        # callback failure occurs immediately after launch.
                        running.setdefault(
                            process,
                            _RunningProcess(
                                index=index,
                                job=job,
                                log_file=log_file,
                                started_at=time.monotonic(),
                            ),
                        )
                    raise

            termination.raise_if_requested()
            completed = [
                (process, returncode)
                for process in running
                if (returncode := process.poll()) is not None
            ]
            termination.raise_if_requested()
            if not completed:
                time.sleep(_POLL_INTERVAL_SECONDS)
                termination.raise_if_requested()
                continue

            for process, returncode in completed:
                termination.raise_if_requested()
                active = running.pop(process)
                active.log_file.close()
                result = ProcessResult(
                    job=active.job,
                    pid=process.pid,
                    returncode=returncode,
                    elapsed_seconds=max(
                        0.0, time.monotonic() - active.started_at
                    ),
                )
                results[active.index] = result
                if on_complete is not None:
                    on_complete(result)
                termination.raise_if_requested()
    except BaseException:
        _stop_processes(
            running,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )
        raise
    finally:
        for active in running.values():
            active.log_file.close()

    # Every slot is populated before the scheduler can exit normally.
    return [result for result in results if result is not None]


def run_process_jobs(
    jobs: Iterable[ProcessJob],
    *,
    max_concurrency: int,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    on_start: ProcessStartCallback | None = None,
    on_complete: ProcessCompleteCallback | None = None,
) -> list[ProcessResult]:
    """Run jobs with bounded concurrency and return results in input order.

    Each job receives a newly created ``output_dir`` and an exclusively created
    log file.  A nonzero child return code is data in ``ProcessResult`` rather
    than an exception.  Launch/filesystem errors and callback exceptions are
    propagated after active process groups are cleaned up.

    SIGTERM and SIGHUP are recorded in the main thread and raised as
    ``ProcessTermination`` at a safe scheduler boundary.  ``KeyboardInterrupt``
    is likewise propagated only after active workers have been stopped.
    """

    if type(max_concurrency) is not int or max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if (
        isinstance(shutdown_grace_seconds, bool)
        or not isinstance(shutdown_grace_seconds, numbers.Real)
        or not math.isfinite(shutdown_grace_seconds)
        or shutdown_grace_seconds < 0
    ):
        raise ValueError(
            "shutdown_grace_seconds must be a finite nonnegative number"
        )

    job_list = list(jobs)
    if not all(isinstance(job, ProcessJob) for job in job_list):
        raise TypeError("jobs must contain only ProcessJob instances")
    if not job_list:
        return []

    with _termination_signal_handlers() as termination:
        return _run_scheduler(
            job_list,
            max_concurrency=min(max_concurrency, len(job_list)),
            shutdown_grace_seconds=float(shutdown_grace_seconds),
            termination=termination,
            on_start=on_start,
            on_complete=on_complete,
        )
