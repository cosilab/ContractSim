"""Shared data models, configuration, and persistence for F1 search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLVER_ROOT = PROJECT_ROOT / "rc-solver"
DEFAULT_ENVIRONMENT_ROOT = PROJECT_ROOT.parent / "envs"
DEFAULT_INITIAL_CONTRACT_PATH = (
    PROJECT_ROOT / "pareto" / "data" / "f1_initial_contract.json"
)
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from contract import Contract
from environment import Environment


SCHEMA_VERSION = 2
INFEASIBLE_SCORE_OFFSET = 1_000_000.0


def _implementation_fingerprint() -> str:
    """Fingerprint every F1 and solver source that can change a trajectory."""
    paths = [
        PROJECT_ROOT / "pareto" / "generation" / filename
        for filename in (
            "f1_frontier.py",
            "f1_common.py",
            "f1_base_search.py",
            "f1_refined_search.py",
        )
    ]
    paths.append(SOLVER_ROOT / "rc_solver.py")
    for directory in ("bellman", "contract", "environment", "runtime", "transitions"):
        paths.extend(
            path
            for path in (SOLVER_ROOT / directory).rglob("*.py")
            if "tests" not in path.parts
        )

    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(PROJECT_ROOT)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


IMPLEMENTATION_FINGERPRINT = _implementation_fingerprint()

COARSE_QUANTITY_DELTAS = (-3, 3)
COARSE_PRICE_DELTAS = (-2, -1, 1, 2)
COARSE_PAYMENT_SWAPS = (10, 20)

FINE_QUANTITY_DELTAS = (-3, -2, -1, 1, 2, 3)
FINE_PRICE_DELTAS = (-1, 1)
FINE_PAYMENT_SWAPS = (1, 5)
START_QUANTITY_MAXIMUMS = {"A": 6, "B": 8, "C": 8}
MIN_RANDOM_START_KEEPERS = 3
MIN_RANDOM_START_SOLVER_CALLS = 80

DEFAULT_MAX_CUSTOMER_GRID = tuple(float(value) for value in range(-40, 201, 10))
DEFAULT_MAX_SUPPLIER_GRID = tuple(float(value) for value in range(180, 421, 10))


class Sweep(str, Enum):
    """Supported epsilon-constraint orientations."""

    MAX_CUSTOMER = "max-customer"
    MAX_SUPPLIER = "max-supplier"

    @property
    def constrained_role(self) -> str:
        return "supplier" if self is Sweep.MAX_CUSTOMER else "customer"

    @property
    def optimized_role(self) -> str:
        return "customer" if self is Sweep.MAX_CUSTOMER else "supplier"


@dataclass(frozen=True)
class F1SearchConfig:
    """Immutable settings that define one reproducible F1 search."""

    sweep: Sweep
    epsilon_grid: tuple[float, ...]
    satisfaction_threshold: float = 0.95
    minimum_satisfaction: float = 0.95
    number_of_starts: int = 3
    coarse_iterations: int = 10
    fine_iterations: int = 50
    prune_margin: float = 20.0
    random_start_attempts: int = 80
    seed: int = 42
    payment_floor: int = 100
    payment_cap: int = 200
    search_mode: str = "best"
    production_weeks: tuple[int, ...] = (2, 4, 6, 8, 10)
    payment_weeks: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
    implementation_fingerprint: str = IMPLEMENTATION_FINGERPRINT

    def __post_init__(self) -> None:
        if self.implementation_fingerprint != IMPLEMENTATION_FINGERPRINT:
            raise ValueError(
                "implementation_fingerprint does not match the running code"
            )
        if not self.epsilon_grid:
            raise ValueError("epsilon_grid cannot be empty")
        if any(not math.isfinite(value) for value in self.epsilon_grid):
            raise ValueError("epsilon_grid values must be finite")
        if tuple(sorted(set(self.epsilon_grid))) != self.epsilon_grid:
            raise ValueError("epsilon_grid must be unique and ascending")
        for name, value in (
            ("satisfaction_threshold", self.satisfaction_threshold),
            ("minimum_satisfaction", self.minimum_satisfaction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.number_of_starts < 1:
            raise ValueError("number_of_starts must be at least one")
        if self.coarse_iterations < 0 or self.fine_iterations < 0:
            raise ValueError("iteration counts must be nonnegative")
        if self.random_start_attempts < 0:
            raise ValueError("random_start_attempts must be nonnegative")
        if self.payment_floor < 0 or self.payment_cap < self.payment_floor:
            raise ValueError("payment bounds are invalid")
        if self.search_mode not in {"best", "first"}:
            raise ValueError("search_mode must be 'best' or 'first'")
        if len(self.payment_weeks) != len(self.production_weeks) + 1:
            raise ValueError("payment_weeks must contain one more week than production_weeks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep": self.sweep.value,
            "epsilon_grid": list(self.epsilon_grid),
            "satisfaction_threshold": self.satisfaction_threshold,
            "minimum_satisfaction": self.minimum_satisfaction,
            "number_of_starts": self.number_of_starts,
            "coarse_iterations": self.coarse_iterations,
            "fine_iterations": self.fine_iterations,
            "prune_margin": self.prune_margin,
            "random_start_attempts": self.random_start_attempts,
            "seed": self.seed,
            "implementation_fingerprint": self.implementation_fingerprint,
            "payment_floor": self.payment_floor,
            "payment_cap": self.payment_cap,
            "search_mode": self.search_mode,
            "production_weeks": list(self.production_weeks),
            "payment_weeks": list(self.payment_weeks),
        }


@dataclass(frozen=True)
class F1RefinedSearchConfig(F1SearchConfig):
    """ContractBench targeted-refinement settings for one anchor-local search."""

    quantity_radius: int = 1
    price_radius: float = 1.0
    payment_radius: int = 5
    refined_iterations: int = 12

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantity_radius < 0 or self.payment_radius < 0:
            raise ValueError("refined quantity and payment radii must be nonnegative")
        if not math.isfinite(self.price_radius) or self.price_radius < 0:
            raise ValueError("refined price radius must be finite and nonnegative")
        if self.refined_iterations < 0:
            raise ValueError("refined_iterations must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "search_variant": "targeted-refinement",
            "quantity_radius": self.quantity_radius,
            "price_radius": self.price_radius,
            "payment_radius": self.payment_radius,
            "refined_iterations": self.refined_iterations,
        }


def _whole_number(value: float, *, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be a whole number; got {value!r}")
    return int(number)


@dataclass(frozen=True)
class ContractCandidate:
    """Hashable integer search representation for a constant-quantity contract."""

    payments: tuple[int, ...]
    quantities: tuple[int, ...]
    prices: tuple[int, ...]

    @classmethod
    def from_contract(
        cls,
        contract: Contract,
        products: tuple[str, ...],
    ) -> "ContractCandidate":
        rows = [
            tuple(int(week.product_quantities[product]) for product in products)
            for week in contract.production_schedule
        ]
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(
                "F1 generation deliberately requires the same product quantities "
                "in every production week"
            )
        return cls(
            payments=tuple(
                _whole_number(week.amount, name=f"payment in week {week.week}")
                for week in contract.payment_schedule
            ),
            quantities=rows[0],
            prices=tuple(
                _whole_number(contract.product_prices[product], name=f"price for {product}")
                for product in products
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractCandidate":
        return cls(
            payments=tuple(int(value) for value in payload["payments"]),
            quantities=tuple(int(value) for value in payload["quantities"]),
            prices=tuple(int(value) for value in payload["prices"]),
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "payments": list(self.payments),
            "quantities": list(self.quantities),
            "prices": list(self.prices),
        }

    def total_payment(self) -> int:
        return sum(self.payments)

    def face_value(self, number_of_rounds: int) -> int:
        return number_of_rounds * sum(
            quantity * price for quantity, price in zip(self.quantities, self.prices)
        )

    def to_contract(
        self,
        products: tuple[str, ...],
        config: F1SearchConfig,
    ) -> Contract:
        quantity_mapping = dict(zip(products, self.quantities))
        return Contract.from_mapping(
            {
                "product_prices": dict(zip(products, self.prices)),
                "production_schedule": [
                    {
                        "week": week,
                        "product_quantities": quantity_mapping,
                    }
                    for week in config.production_weeks
                ],
                "payment_schedule": [
                    {"week": week, "amount": amount}
                    for week, amount in zip(config.payment_weeks, self.payments)
                ],
            }
        )


@dataclass(frozen=True)
class RefinedContractCandidate:
    """Targeted-refinement state with one quantity vector per round."""

    payments: tuple[int, ...]
    quantity_schedule: tuple[tuple[int, ...], ...]
    prices: tuple[float, ...]

    @classmethod
    def from_contract(
        cls,
        contract: Contract,
        products: tuple[str, ...],
    ) -> "RefinedContractCandidate":
        return cls(
            payments=tuple(
                _whole_number(week.amount, name=f"payment in week {week.week}")
                for week in contract.payment_schedule
            ),
            quantity_schedule=tuple(
                tuple(int(week.product_quantities[product]) for product in products)
                for week in contract.production_schedule
            ),
            prices=tuple(float(contract.product_prices[product]) for product in products),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefinedContractCandidate":
        return cls(
            payments=tuple(int(value) for value in payload["payments"]),
            quantity_schedule=tuple(
                tuple(int(value) for value in row)
                for row in payload["quantity_schedule"]
            ),
            prices=tuple(float(value) for value in payload["prices"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "payments": list(self.payments),
            "quantity_schedule": [list(row) for row in self.quantity_schedule],
            "prices": list(self.prices),
        }

    def total_payment(self) -> int:
        return sum(self.payments)

    def face_value(self) -> float:
        return sum(
            quantity * self.prices[index]
            for row in self.quantity_schedule
            for index, quantity in enumerate(row)
        )

    def to_contract(
        self,
        products: tuple[str, ...],
        config: F1SearchConfig,
    ) -> Contract:
        return Contract.from_mapping(
            {
                "product_prices": dict(zip(products, self.prices)),
                "production_schedule": [
                    {
                        "week": week,
                        "product_quantities": dict(zip(products, quantities)),
                    }
                    for week, quantities in zip(
                        config.production_weeks, self.quantity_schedule
                    )
                ],
                "payment_schedule": [
                    {"week": week, "amount": amount}
                    for week, amount in zip(config.payment_weeks, self.payments)
                ],
            }
        )


SearchCandidate = Union[ContractCandidate, RefinedContractCandidate]


def _candidate_from_dict(payload: Mapping[str, Any]) -> SearchCandidate:
    if "quantity_schedule" in payload:
        return RefinedContractCandidate.from_dict(payload)
    return ContractCandidate.from_dict(payload)


@dataclass(frozen=True)
class Evaluation:
    satisfaction_probability: float
    supplier_utility: float
    customer_utility: float

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> "Evaluation":
        return cls(
            satisfaction_probability=float(result["P_sat"]),
            supplier_utility=float(result["E_supplier_utility"]),
            customer_utility=float(result["E_customer_utility"]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "P_sat": self.satisfaction_probability,
            "E_supplier_utility": self.supplier_utility,
            "E_customer_utility": self.customer_utility,
        }


@dataclass(frozen=True)
class FrontierPoint:
    epsilon: float
    sweep: Sweep
    candidate: SearchCandidate
    evaluation: Evaluation
    constraint_met: bool
    score: float | None
    trace: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrontierPoint":
        return cls(
            epsilon=float(payload["epsilon"]),
            sweep=Sweep(payload["sweep"]),
            candidate=_candidate_from_dict(payload["candidate"]),
            evaluation=Evaluation(
                satisfaction_probability=float(payload["P_sat"]),
                supplier_utility=float(payload["E_supplier_utility"]),
                customer_utility=float(payload["E_customer_utility"]),
            ),
            constraint_met=bool(payload["constraint_met"]),
            score=None if payload.get("score") is None else float(payload["score"]),
            trace=tuple(payload.get("trace", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "epsilon": self.epsilon,
            "sweep": self.sweep.value,
            "epsilon_role": self.sweep.constrained_role,
            "optimized_role": self.sweep.optimized_role,
            "candidate": self.candidate.to_dict(),
            **self.evaluation.to_dict(),
            "constraint_met": self.constraint_met,
            "score": self.score,
        }
        if self.trace:
            payload["trace"] = list(self.trace)
        return payload


@dataclass
class SearchCheckpoint:
    """Durable progress for one environment and sweep direction."""

    config_fingerprint: str
    environment_fingerprint: str
    initial_candidate: SearchCandidate
    completed_points: list[FrontierPoint]
    epsilon_index: int = 0
    start_index: int = 0
    current_starts: tuple[SearchCandidate, ...] = ()
    current_best: FrontierPoint | None = None
    warm_start: SearchCandidate | None = None
    status: str = "running"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchCheckpoint":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported F1 checkpoint schema")
        return cls(
            config_fingerprint=str(payload["config_fingerprint"]),
            environment_fingerprint=str(payload["environment_fingerprint"]),
            initial_candidate=_candidate_from_dict(payload["initial_candidate"]),
            completed_points=[
                FrontierPoint.from_dict(point) for point in payload["completed_points"]
            ],
            epsilon_index=int(payload["epsilon_index"]),
            start_index=int(payload["start_index"]),
            current_starts=tuple(
                _candidate_from_dict(candidate)
                for candidate in payload.get("current_starts", [])
            ),
            current_best=(
                None
                if payload.get("current_best") is None
                else FrontierPoint.from_dict(payload["current_best"])
            ),
            warm_start=(
                None
                if payload.get("warm_start") is None
                else _candidate_from_dict(payload["warm_start"])
            ),
            status=str(payload.get("status", "running")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "config_fingerprint": self.config_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "initial_candidate": self.initial_candidate.to_dict(),
            "completed_points": [point.to_dict() for point in self.completed_points],
            "epsilon_index": self.epsilon_index,
            "start_index": self.start_index,
            "current_starts": [candidate.to_dict() for candidate in self.current_starts],
            "current_best": None if self.current_best is None else self.current_best.to_dict(),
            "warm_start": None if self.warm_start is None else self.warm_start.to_dict(),
        }


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_environment_path(
    *,
    environment_id: int | None,
    environment_path: str | Path | None,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> Path:
    """Resolve exactly one CLI environment selector to a JSON file."""
    if (environment_id is None) == (environment_path is None):
        raise ValueError("Select exactly one of --env or --environment")

    if environment_id is not None:
        if (
            isinstance(environment_id, bool)
            or not isinstance(environment_id, int)
            or environment_id < 1
        ):
            raise ValueError(
                f"Environment ID must be a positive integer; got {environment_id!r}"
            )
        selected = (
            Path(environment_root)
            / str(environment_id)
            / "environment.json"
        )
    else:
        assert environment_path is not None
        selected = Path(environment_path).expanduser()

    resolved = selected.resolve()
    if not resolved.is_file():
        if environment_id is None:
            raise FileNotFoundError(f"Missing environment JSON: {resolved}")
        raise FileNotFoundError(
            f"Missing environment JSON for environment ID {environment_id}: "
            f"{resolved}"
        )
    return resolved


def load_environment_selection(
    *,
    environment_id: int | None,
    environment_path: str | Path | None,
    environment_root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
) -> tuple[Path, Environment]:
    """Instantiate the data-driven environment selected on the CLI."""
    resolved = resolve_environment_path(
        environment_id=environment_id,
        environment_path=environment_path,
        environment_root=environment_root,
    )
    environment = Environment.from_json(resolved)
    if (
        environment_id is not None
        and environment.environment_id != environment_id
    ):
        raise ValueError(
            f"Environment file for ID {environment_id} declares ID "
            f"{environment.environment_id}: {resolved}"
        )
    return resolved, environment


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> SearchCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"Missing F1 checkpoint: {path}")
    return SearchCheckpoint.from_dict(json.loads(path.read_text()))


def save_checkpoint_atomic(path: Path, checkpoint: SearchCheckpoint) -> None:
    save_json_atomic(path, checkpoint.to_dict())


def parse_epsilon_grid(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epsilon grid must contain numbers") from exc
    try:
        F1SearchConfig(sweep=Sweep.MAX_CUSTOMER, epsilon_grid=values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return values
