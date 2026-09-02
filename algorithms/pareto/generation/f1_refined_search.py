"""ContractBench-compatible anchor-local refined F1 search."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from .f1_base_search import F1FrontierSearch
from .f1_common import (
    F1RefinedSearchConfig,
    FrontierPoint,
    RefinedContractCandidate,
    SearchCandidate,
    load_checkpoint,
    save_checkpoint_atomic,
)

if TYPE_CHECKING:
    from environment import Environment
    from rc_solver import RCCSolver


class F1FrontierSearchRefined(F1FrontierSearch):
    """Modular-solver port of ContractBench's targeted refinement."""

    def __init__(
        self,
        solver: RCCSolver,
        environment: Environment,
        config: F1RefinedSearchConfig,
        initial_candidate: RefinedContractCandidate,
        checkpoint_path: Path,
        *,
        environment_fingerprint: str,
    ) -> None:
        if not isinstance(config, F1RefinedSearchConfig):
            raise TypeError("refined search requires F1RefinedSearchConfig")
        if not isinstance(initial_candidate, RefinedContractCandidate):
            raise TypeError("refined search requires RefinedContractCandidate")
        self.anchor_candidate = initial_candidate
        super().__init__(
            solver,
            environment,
            config,
            initial_candidate,
            checkpoint_path,
            environment_fingerprint=environment_fingerprint,
        )

    @property
    def refined_config(self) -> F1RefinedSearchConfig:
        assert isinstance(self.config, F1RefinedSearchConfig)
        return self.config

    def _within_local_box(self, candidate: RefinedContractCandidate) -> bool:
        config = self.refined_config
        anchor = self.anchor_candidate
        if any(
            abs(value - origin) > config.payment_radius
            for value, origin in zip(candidate.payments, anchor.payments)
        ):
            return False
        if any(
            abs(value - origin) > config.quantity_radius
            for row, anchor_row in zip(
                candidate.quantity_schedule, anchor.quantity_schedule
            )
            for value, origin in zip(row, anchor_row)
        ):
            return False
        return not any(
            abs(value - origin) > config.price_radius + 1e-9
            for value, origin in zip(candidate.prices, anchor.prices)
        )

    def is_feasible(self, candidate: SearchCandidate) -> bool:
        if not isinstance(candidate, RefinedContractCandidate):
            return False
        if len(candidate.payments) != len(self.config.payment_weeks):
            return False
        if len(candidate.quantity_schedule) != len(self.config.production_weeks):
            return False
        if len(candidate.prices) != len(self.products):
            return False
        if any(len(row) != len(self.products) for row in candidate.quantity_schedule):
            return False
        if not self._within_local_box(candidate):
            return False
        if any(payment < 0 for payment in candidate.payments):
            return False
        if not (
            self.config.payment_floor
            <= candidate.total_payment()
            <= self.config.payment_cap
        ):
            return False
        if not math.isclose(
            candidate.total_payment(), candidate.face_value(), abs_tol=1e-7
        ):
            return False
        for row in candidate.quantity_schedule:
            if any(
                quantity < minimum
                for quantity, minimum in zip(row, self.minimum_quantities)
            ):
                return False
            if not self._needs_fit(row):
                return False
        # This deliberately matches targeted_refine.py: the base PRICE_MIN
        # and private-value ceiling are not contract-validity constraints here.
        return not any(price < 0.0 for price in candidate.prices)

    def point(
        self,
        candidate: SearchCandidate,
        epsilon: float,
    ) -> FrontierPoint:
        if not isinstance(candidate, RefinedContractCandidate):
            raise TypeError("refined search can only evaluate refined candidates")
        return super().point(candidate, epsilon)

    def _bounded_payment_distributions(
        self,
        payments: tuple[int, ...],
        shift: int,
    ) -> Iterator[tuple[int, ...]]:
        """Match ContractBench's deterministic exact-sum adjustments in order."""
        if shift == 0:
            yield payments
            return

        anchor_payments = self.anchor_candidate.payments
        radius = self.refined_config.payment_radius
        seen: set[tuple[int, ...]] = set()

        def emit(order: Iterable[int]) -> tuple[int, ...] | None:
            updated = list(payments)
            remaining = abs(int(shift))
            direction = 1 if shift > 0 else -1
            for index in order:
                bound = (
                    anchor_payments[index] + radius
                    if direction > 0
                    else max(0, anchor_payments[index] - radius)
                )
                capacity = (
                    bound - updated[index]
                    if direction > 0
                    else updated[index] - bound
                )
                amount = min(remaining, max(0, capacity))
                updated[index] += direction * amount
                remaining -= amount
                if remaining == 0:
                    candidate = tuple(updated)
                    if candidate not in seen:
                        seen.add(candidate)
                        return candidate
                    return None
            return None

        count = len(payments)
        orders = [tuple(range(count)), tuple(reversed(range(count)))]
        orders.extend(
            tuple((offset + index) % count for index in range(count))
            for offset in range(count)
        )
        for order in orders:
            candidate = emit(order)
            if candidate is not None:
                yield candidate

    @staticmethod
    def _refined_key(candidate: RefinedContractCandidate) -> tuple[Any, ...]:
        return (
            candidate.payments,
            candidate.quantity_schedule,
            tuple(round(price, 8) for price in candidate.prices),
        )

    def _refined_neighbors(
        self,
        candidate: RefinedContractCandidate,
    ) -> Iterator[tuple[str, RefinedContractCandidate]]:
        seen: set[tuple[Any, ...]] = set()

        def maybe_emit(
            label: str,
            neighbor: RefinedContractCandidate,
        ) -> Iterator[tuple[str, RefinedContractCandidate]]:
            key = self._refined_key(neighbor)
            if key in seen or not self.is_feasible(neighbor):
                return
            seen.add(key)
            yield label, neighbor

        rounds = len(self.config.production_weeks)
        payments_count = len(self.config.payment_weeks)

        # Move production between rounds without changing face value.
        for product_index, product in enumerate(self.products):
            for source_round in range(rounds):
                for target_round in range(rounds):
                    if source_round == target_round:
                        continue
                    schedule = [list(row) for row in candidate.quantity_schedule]
                    schedule[source_round][product_index] -= 1
                    schedule[target_round][product_index] += 1
                    neighbor = RefinedContractCandidate(
                        payments=candidate.payments,
                        quantity_schedule=tuple(tuple(row) for row in schedule),
                        prices=candidate.prices,
                    )
                    yield from maybe_emit(
                        f"move_{product}_r{source_round + 1}_to_r{target_round + 1}",
                        neighbor,
                    )

        # Change one round quantity and rebalance payments to the new face value.
        for round_index in range(rounds):
            for product_index, product in enumerate(self.products):
                for delta in (-1, 1):
                    schedule = [list(row) for row in candidate.quantity_schedule]
                    schedule[round_index][product_index] += delta
                    provisional = RefinedContractCandidate(
                        payments=candidate.payments,
                        quantity_schedule=tuple(tuple(row) for row in schedule),
                        prices=candidate.prices,
                    )
                    new_face_value = provisional.face_value()
                    if not math.isclose(
                        new_face_value, round(new_face_value), abs_tol=1e-7
                    ):
                        continue
                    shift = int(round(new_face_value)) - candidate.total_payment()
                    for payments in self._bounded_payment_distributions(
                        candidate.payments, shift
                    ):
                        neighbor = RefinedContractCandidate(
                            payments=payments,
                            quantity_schedule=provisional.quantity_schedule,
                            prices=candidate.prices,
                        )
                        yield from maybe_emit(
                            f"qty_{product}_r{round_index + 1}{delta:+d}",
                            neighbor,
                        )

        # Change one unit price and rebalance payments. Fractional anchors keep
        # their fractional component, as in ContractBench's refined search.
        for product_index, product in enumerate(self.products):
            total_quantity = sum(
                row[product_index] for row in candidate.quantity_schedule
            )
            for delta in (-1.0, 1.0):
                prices = list(candidate.prices)
                prices[product_index] += delta
                shift = int(round(total_quantity * delta))
                for payments in self._bounded_payment_distributions(
                    candidate.payments, shift
                ):
                    neighbor = RefinedContractCandidate(
                        payments=payments,
                        quantity_schedule=candidate.quantity_schedule,
                        prices=tuple(prices),
                    )
                    yield from maybe_emit(f"price_{product}{delta:+g}", neighbor)

        # Redistribute fixed-total payment timing inside the local box.
        for source in range(payments_count):
            for target in range(payments_count):
                if source == target:
                    continue
                for amount in (1, 5):
                    payments = list(candidate.payments)
                    payments[source] -= amount
                    payments[target] += amount
                    neighbor = RefinedContractCandidate(
                        payments=tuple(payments),
                        quantity_schedule=candidate.quantity_schedule,
                        prices=candidate.prices,
                    )
                    yield from maybe_emit(
                        f"pay_{source + 1}_to_{target + 1}_{amount}", neighbor
                    )

    def _local_ascent(
        self,
        initial: RefinedContractCandidate,
        epsilon: float,
    ) -> FrontierPoint:
        current = self.point(initial, epsilon)
        trace: list[dict[str, Any]] = []

        for iteration in range(self.refined_config.refined_iterations):
            best: tuple[str, FrontierPoint] | None = None
            evaluations = 0
            for label, candidate in self._refined_neighbors(current.candidate):
                point = self.point(candidate, epsilon)
                evaluations += 1
                if self._score(point) > self._score(current) + 1e-9 and (
                    best is None or self._score(point) > self._score(best[1]) + 1e-9
                ):
                    best = (label, point)
                    if self.config.search_mode == "first":
                        break

            if best is None:
                trace.append(
                    {
                        "iteration": iteration,
                        "move": None,
                        "evaluations": evaluations,
                        "status": "converged",
                    }
                )
                break

            label, current = best
            trace.append(
                {
                    "iteration": iteration,
                    "move": label,
                    "evaluations": evaluations,
                    "score": current.score,
                    **current.evaluation.to_dict(),
                }
            )
            print(
                f"  iter={iteration:02d} move={label} evals={evaluations} "
                f"P={current.evaluation.satisfaction_probability:.4f} "
                f"UK={current.evaluation.supplier_utility:.3f} "
                f"UC={current.evaluation.customer_utility:.3f}",
                flush=True,
            )

        return replace(current, trace=tuple(trace))

    def run(self, *, resume: bool = False) -> list[FrontierPoint]:
        checkpoint = load_checkpoint(self.checkpoint_path) if resume else self._new_checkpoint()
        self._validate_checkpoint(checkpoint)
        if checkpoint.status == "complete":
            return list(checkpoint.completed_points)
        if not resume:
            save_checkpoint_atomic(self.checkpoint_path, checkpoint)

        while checkpoint.epsilon_index < len(self.config.epsilon_grid):
            epsilon = self.config.epsilon_grid[checkpoint.epsilon_index]
            seed = checkpoint.warm_start or self.initial_candidate
            if not isinstance(seed, RefinedContractCandidate):
                raise ValueError("Refined checkpoint contains a base-search candidate")
            point = self._local_ascent(seed, epsilon)
            checkpoint.completed_points.append(point)
            checkpoint.warm_start = point.candidate
            checkpoint.epsilon_index += 1
            save_checkpoint_atomic(self.checkpoint_path, checkpoint)

        checkpoint.status = "complete"
        save_checkpoint_atomic(self.checkpoint_path, checkpoint)
        return list(checkpoint.completed_points)

    def result_payload(self, points: list[FrontierPoint]) -> dict[str, Any]:
        return {
            **super().result_payload(points),
            "search_variant": "targeted-refinement",
            "local_box_anchor": self.anchor_candidate.to_dict(),
        }
