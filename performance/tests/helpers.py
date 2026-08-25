from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sample_terms() -> dict:
    return {
        "dish_prices": {"A": 10, "B": 6, "C": 3},
        "payment_schedule": [
            {"week": week, "amount": 10}
            for week in (1, 3, 5, 7, 9, 11)
        ],
        "production_schedule": [
            {"week": week, "A": 1, "B": 1, "C": 1}
            for week in (2, 4, 6, 8, 10)
        ],
    }


def sample_config(environment_id: int = 1) -> dict:
    return {
        "environment_id": environment_id,
        "contingencies": [],
        "contingency_mode": "grim_trigger",
    }


def write_policy(contract_dir: Path) -> Path:
    path = contract_dir / "baselines" / "rcc" / "policy.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        policies=np.array([2, 2, 2], dtype=np.int64),
        policy_axes=np.array(json.dumps(["ingredient"])),
        PRICE_RANGES=np.array([[1, 1, 2]], dtype=np.int64),
    )
    return path


def write_contract(contract_dir: Path, environment_id: int = 1) -> Path:
    contract_dir.mkdir(parents=True)
    (contract_dir / "contract.txt").write_text(
        "A simple synthetic catering contract.",
        encoding="utf-8",
    )
    (contract_dir / "terms.json").write_text(
        json.dumps(sample_terms()),
        encoding="utf-8",
    )
    (contract_dir / "config.json").write_text(
        json.dumps(sample_config(environment_id)),
        encoding="utf-8",
    )
    write_policy(contract_dir)
    return contract_dir
