"""Shared test fixture paths and loaders."""

from pathlib import Path

from environment.json_input import load_json_object


DATA_DIR = Path(__file__).resolve().parent / "data"
ENVIRONMENT_09_PATH = DATA_DIR / "environment_09.json"
ROLLOVER_CONTRACT_PATH = DATA_DIR / "rollover_contract.json"


def load_environment_09() -> dict:
    """Return a fresh copy of the canonical environment 9 test fixture."""
    return load_json_object(ENVIRONMENT_09_PATH, name="Environment fixture")


def load_rollover_contract() -> dict:
    """Return a fresh copy of the rollover contract fixture."""
    return load_json_object(ROLLOVER_CONTRACT_PATH, name="Rollover contract JSON")
