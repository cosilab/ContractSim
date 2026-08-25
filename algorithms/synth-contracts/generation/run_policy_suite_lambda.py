#!/usr/bin/env python3
"""Validate a Lambda GPU host, test policy export, and generate the suite."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RC_SOLVER_ROOT = REPOSITORY_ROOT / "algorithms" / "rc-solver"
GENERATOR = Path(__file__).with_name("generate_policy_suite.py")
OUTPUT_ROOT = REPOSITORY_ROOT / "algorithms" / "synth-contracts" / "data" / "policy-suite"
OUTPUT_ARCHIVE = REPOSITORY_ROOT.parent / "policy-suite-output"
MINIMUM_FREE_BYTES = 200 * 1024**3


def main() -> None:
    try:
        import jax
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            f"Missing required preinstalled dependency: {exc.name}. "
            "Nothing was installed."
        ) from exc

    devices = jax.devices()
    print(f"Python {sys.version.split()[0]}", flush=True)
    print(f"JAX {jax.__version__}; NumPy {np.__version__}", flush=True)
    print(f"JAX devices: {devices}", flush=True)
    if not any(device.platform == "gpu" for device in devices):
        raise SystemExit("JAX does not see a GPU; generation was not started.")

    free_bytes = shutil.disk_usage(REPOSITORY_ROOT).free
    print(f"Free disk: {free_bytes / 1024**3:.1f} GiB", flush=True)
    if free_bytes < MINIMUM_FREE_BYTES:
        raise SystemExit(
            "At least 200 GiB of free disk is required for the worst-case "
            "uncompressed policy output; generation was not started."
        )

    print("Running policy-export tests and eight-mode smoke test...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "tests.test_contingency_policy_export",
        ],
        cwd=RC_SOLVER_ROOT,
        check=True,
    )

    print("Tests passed. Starting/resuming the 240-policy suite...", flush=True)
    subprocess.run(
        [sys.executable, str(GENERATOR), "--resume"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    archive = shutil.make_archive(
        str(OUTPUT_ARCHIVE),
        "zip",
        root_dir=OUTPUT_ROOT.parent,
        base_dir=OUTPUT_ROOT.name,
    )
    print(f"Done: {archive}", flush=True)


if __name__ == "__main__":
    main()
