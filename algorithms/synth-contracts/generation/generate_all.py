#!/usr/bin/env python3
"""Generate base contracts and the complete two-GPU policy suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GENERATION_DIR = Path(__file__).resolve().parent
BASE_GENERATOR = GENERATION_DIR / "generate.py"
POLICY_GENERATOR = GENERATION_DIR / "generate_policy_suite.py"


def _run_on_gpu(gpu: str, commands: list[list[str]]) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for command in commands:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )


def _run_groups(groups: list[tuple[str, list[list[str]]]]) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_run_on_gpu, gpu, commands)
            for gpu, commands in groups
        ]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise SystemExit(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    base_root = run_dir / "base"
    suite_root = run_dir / "suite"
    resume = ["--resume"] if args.resume else []

    base_groups = []
    for gpu, environment_ids in zip(args.gpus, ((3, 4), (5, 6))):
        commands = [
            [
                sys.executable,
                str(BASE_GENERATOR),
                "--env",
                str(environment_id),
                "--out-dir",
                str(base_root),
                *resume,
            ]
            for environment_id in environment_ids
        ]
        base_groups.append((gpu, commands))
    _run_groups(base_groups)

    with tempfile.TemporaryDirectory(dir=run_dir) as temporary_directory:
        temporary_root = Path(temporary_directory)
        policy_groups = []
        policy_environment_groups = ((1, 3, 4), (2, 5, 6))
        manifest_paths = []
        for index, (gpu, environment_ids) in enumerate(
            zip(args.gpus, policy_environment_groups)
        ):
            manifest_path = temporary_root / f"manifest-{index}.json"
            manifest_paths.append(manifest_path)
            command = [
                sys.executable,
                str(POLICY_GENERATOR),
                "--source-root",
                str(base_root),
                "--output-root",
                str(suite_root),
                "--evaluation-envs",
                *(str(environment_id) for environment_id in environment_ids),
                "--manifest-path",
                str(manifest_path),
                *resume,
            ]
            policy_groups.append((gpu, [command]))
        _run_groups(policy_groups)

        manifest = []
        for path in manifest_paths:
            manifest.extend(json.loads(path.read_text(encoding="utf-8")))
        manifest.sort(key=lambda entry: entry["instance_id"])
        (suite_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"Generated {len(manifest)} contracts in {suite_root}")


if __name__ == "__main__":
    main()
