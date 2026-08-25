"""Strict, atomic artifact writing and exclusive run-directory claims."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


PRIVATE_FILE_MODE = 0o600
PUBLIC_FILE_MODE = 0o644
RUN_STATUS_FILENAME = "run_status.json"
RUN_STATUS_SCHEMA_VERSION = 1

__all__ = (
    "PRIVATE_FILE_MODE",
    "PUBLIC_FILE_MODE",
    "RUN_STATUS_FILENAME",
    "RUN_STATUS_SCHEMA_VERSION",
    "RunClaim",
    "RunClaimError",
    "claim_output_directory",
    "json_safe",
    "write_json",
    "write_text",
)


class RunClaimError(RuntimeError):
    """Raised when a run directory cannot be claimed or finalized safely."""


def json_safe(value: Any) -> Any:
    """Convert supported runtime values into strict JSON-native values.

    Opaque objects and non-finite numbers are rejected instead of being
    stringified. Mapping keys retain the established artifact behavior of being
    normalized to strings.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Run artifacts cannot contain non-finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} into a run artifact")


def _file_mode(mode: int) -> int:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError("artifact file mode must be an integer")
    if mode < 0 or mode > 0o777:
        raise ValueError("artifact file mode must contain only permission bits")
    return mode


def _json_document(payload: Any) -> str:
    return (
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _open_exclusive(path: Path, mode: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return descriptor


def _write_descriptor(descriptor: int, text: str) -> None:
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def write_text(
    path: str | Path,
    text: str,
    *,
    mode: int = PRIVATE_FILE_MODE,
) -> Path:
    """Atomically write UTF-8 text with an explicit final file mode.

    Private mode is the safe default. Callers writing intentionally public
    benchmark artifacts can pass ``mode=PUBLIC_FILE_MODE`` to retain the
    established ``0644`` behavior.
    """
    if not isinstance(text, str):
        raise TypeError("artifact text must be a string")
    final_mode = _file_mode(mode)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor: int | None = None
    try:
        descriptor = _open_exclusive(temporary, final_mode)
        active_descriptor = descriptor
        descriptor = None
        _write_descriptor(active_descriptor, text)
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def write_json(
    path: str | Path,
    payload: Any,
    *,
    mode: int = PRIVATE_FILE_MODE,
) -> Path:
    """Atomically write one strict, indented UTF-8 JSON document."""
    return write_text(path, _json_document(payload), mode=mode)


def _status_document(
    status: str,
    payload: Mapping[str, Any] | None,
    *,
    schema_version: int,
) -> dict[str, Any]:
    if not isinstance(status, str) or not status or status.strip() != status:
        raise ValueError("run status must be a non-empty trimmed string")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("run status schema_version must be an integer")
    if schema_version < 1:
        raise ValueError("run status schema_version must be positive")
    if payload is None:
        details: dict[str, Any] = {}
    elif isinstance(payload, Mapping):
        converted = json_safe(payload)
        assert isinstance(converted, dict)
        details = converted
    else:
        raise TypeError("run status payload must be a mapping")

    declared_schema = details.pop("schema_version", schema_version)
    if declared_schema != schema_version:
        raise ValueError(
            "run status payload schema_version does not match the claim"
        )
    declared_status = details.pop("status", status)
    if declared_status != status:
        raise ValueError("run status payload does not match the final status")
    return {
        "schema_version": schema_version,
        "status": status,
        **details,
    }


@dataclass
class RunClaim:
    """Exclusive ownership of one output directory's run-status document.

    Use as a context manager whenever possible. A normal context exit completes
    an otherwise-running claim; an exceptional exit atomically records an error
    and never suppresses the original exception.
    """

    output_dir: Path
    status_path: Path
    mode: int = PRIVATE_FILE_MODE
    schema_version: int = RUN_STATUS_SCHEMA_VERSION
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    _terminal_status: str | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def terminal_status(self) -> str | None:
        return self._terminal_status

    @property
    def finalized(self) -> bool:
        return self._terminal_status is not None

    def _finalize(
        self,
        status: str,
        payload: Mapping[str, Any] | None,
        *,
        replace_terminal: bool = False,
    ) -> Path:
        if status == "running":
            raise ValueError("running is not a terminal run status")
        if payload is None:
            combined_payload: Mapping[str, Any] | None = self._metadata
        elif isinstance(payload, Mapping):
            combined_payload = {**self._metadata, **payload}
        else:
            combined_payload = payload
        document = _status_document(
            status,
            combined_payload,
            schema_version=self.schema_version,
        )
        with self._lock:
            if self._terminal_status is not None and not replace_terminal:
                raise RunClaimError(
                    f"run claim was already finalized as {self._terminal_status!r}"
                )
            write_json(self.status_path, document, mode=self.mode)
            self._terminal_status = status
        return self.status_path

    def finalize(
        self,
        status: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically replace ``running`` with a caller-defined terminal status."""
        return self._finalize(status, payload)

    def complete(self, payload: Mapping[str, Any] | None = None) -> Path:
        """Finalize the run with status ``complete``."""
        return self._finalize("complete", payload)

    def fail(
        self,
        error: BaseException | str,
        payload: Mapping[str, Any] | None = None,
        *,
        _replace_terminal: bool = False,
    ) -> Path:
        """Finalize the run with a compact error type and message."""
        if not isinstance(error, (BaseException, str)):
            raise TypeError("run error must be an exception or string")
        if payload is None:
            details: dict[str, Any] = {}
        elif isinstance(payload, Mapping):
            details = dict(payload)
        else:
            raise TypeError("run status payload must be a mapping")
        if isinstance(error, BaseException):
            error_type = type(error).__name__
            message = str(error)
        else:
            error_type = "RunError"
            message = error
        details["error"] = {"type": error_type, "message": message}
        return self._finalize(
            "error",
            details,
            replace_terminal=_replace_terminal,
        )

    def __enter__(self) -> RunClaim:
        if self.finalized:
            raise RunClaimError("cannot enter an already-finalized run claim")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, traceback
        if exception is not None:
            try:
                self.fail(exception, _replace_terminal=True)
            except BaseException as finalization_error:
                if hasattr(exception, "add_note"):
                    exception.add_note(
                        "Additionally, the run error status could not be written: "
                        f"{finalization_error}"
                    )
            return False
        if not self.finalized:
            try:
                self.complete()
            except BaseException as completion_error:
                try:
                    self.fail(completion_error, _replace_terminal=True)
                except BaseException as finalization_error:
                    if hasattr(completion_error, "add_note"):
                        completion_error.add_note(
                            "Additionally, the run error status could not be written: "
                            f"{finalization_error}"
                        )
                raise
        return False


def claim_output_directory(
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    status_filename: str = RUN_STATUS_FILENAME,
    mode: int = PRIVATE_FILE_MODE,
    schema_version: int = RUN_STATUS_SCHEMA_VERSION,
) -> RunClaim:
    """Exclusively claim an output directory using an ``O_EXCL`` status file.

    Scheduler-owned files such as ``stdout.log`` may already be present. The
    status file alone is the ownership sentinel; it is never overwritten by a
    second claimant.
    """
    if (
        not isinstance(status_filename, str)
        or not status_filename
        or Path(status_filename).name != status_filename
        or status_filename in {".", ".."}
    ):
        raise ValueError("status_filename must be one plain filename")
    final_mode = _file_mode(mode)
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise RunClaimError(f"output path is not a directory: {directory}")
    status_path = directory / status_filename
    document = _status_document(
        "running",
        metadata,
        schema_version=schema_version,
    )
    serialized = _json_document(document)
    directory.mkdir(parents=True, exist_ok=True)

    descriptor: int | None = None
    created = False
    try:
        descriptor = _open_exclusive(status_path, final_mode)
        created = True
        active_descriptor = descriptor
        descriptor = None
        _write_descriptor(active_descriptor, serialized)
    except FileExistsError as exc:
        raise RunClaimError(
            f"output directory was claimed by another run: {directory}"
        ) from exc
    except BaseException:
        if created:
            try:
                status_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)

    return RunClaim(
        output_dir=directory,
        status_path=status_path,
        mode=final_mode,
        schema_version=schema_version,
        _metadata={
            key: value
            for key, value in document.items()
            if key not in {"schema_version", "status"}
        },
    )
