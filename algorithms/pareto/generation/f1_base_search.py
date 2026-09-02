"""ContractBench-compatible base F1 epsilon search."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .f1_common import (
    COARSE_PAYMENT_SWAPS,
    COARSE_PRICE_DELTAS,
    COARSE_QUANTITY_DELTAS,
    FINE_PAYMENT_SWAPS,
    FINE_PRICE_DELTAS,
    FINE_QUANTITY_DELTAS,
    INFEASIBLE_SCORE_OFFSET,
    MIN_RANDOM_START_KEEPERS,
    MIN_RANDOM_START_SOLVER_CALLS,
    SCHEMA_VERSION,
    START_QUANTITY_MAXIMUMS,
    ContractCandidate,
    Evaluation,
    F1SearchConfig,
    FrontierPoint,
    SearchCandidate,
    SearchCheckpoint,
    Sweep,
    _json_fingerprint,
    load_checkpoint,
    save_checkpoint_atomic,
)

if TYPE_CHECKING:
    from environment import Environment
    from rc_solver import RCCSolver


class F1FrontierSearch:
    """Stateful multi-start epsilon search around one reusable modular solver."""

    def __init__(
        self,
        solver: RCCSolver,
        environment: Environment,
        config: F1SearchConfig,
        initial_candidate: SearchCandidate,
        checkpoint_path: Path,
        *,
        environment_fingerprint: str,
    ) -> None:
        self.solver = solver
        self.environment = environment
        self.config = config
        self.initial_candidate = initial_candidate
        self.checkpoint_path = checkpoint_path
        self.environment_fingerprint = environment_fingerprint
        self.products = environment.supplier.products
        self.minimum_quantities = tuple(
            environment.customer.minimum_products_per_production_round[product]
            for product in self.products
        )
        if isinstance(initial_candidate, ContractCandidate):
            self.price_minimums = self._price_minimums()
            self.price_maximums = tuple(
                environment.customer.private_values[product]
                for product in self.products
            )
        else:
            self.price_minimums = ()
            self.price_maximums = ()
        self.random_quantity_maximums = (
            self._random_quantity_maximums()
            if isinstance(initial_candidate, ContractCandidate)
            and config.number_of_starts > 1
            else ()
        )
        self.evaluation_cache: dict[SearchCandidate, Evaluation] = {}
        config_payload = config.to_dict()
        self.config_fingerprint = _json_fingerprint(config_payload)

        if len(initial_candidate.payments) != len(config.payment_weeks):
            raise ValueError("Initial candidate payment count does not match payment_weeks")
        candidate_product_count = (
            len(initial_candidate.quantities)
            if isinstance(initial_candidate, ContractCandidate)
            else len(initial_candidate.prices)
        )
        if candidate_product_count != len(self.products):
            raise ValueError("Initial candidate quantities do not match environment products")
        if config.payment_cap > int(environment.customer.starting_budget):
            raise ValueError("payment_cap cannot exceed the customer's starting budget")
        if not self.is_feasible(initial_candidate):
            raise ValueError(f"Initial F1 candidate is infeasible: {initial_candidate}")

    def _price_minimums(self) -> tuple[int, ...]:
        outcomes = self.environment.probability_dist.price_distribution.outcomes_by_resource
        midpoint = {
            resource: (rows[0].value + rows[-1].value) / 2
            for resource, rows in outcomes.items()
        }
        return tuple(
            int(
                sum(
                    quantity * midpoint[resource]
                    for resource, quantity in self.environment.supplier.recipes[product].items()
                )
            )
            for product in self.products
        )

    def _random_quantity_maximums(self) -> tuple[int, ...]:
        expected_products = tuple(START_QUANTITY_MAXIMUMS)
        if self.products != expected_products:
            raise ValueError(
                "F1 random starts require products in A, B, C order"
            )
        if len(self.config.production_weeks) != 5 or len(self.config.payment_weeks) != 6:
            raise ValueError(
                "F1 random starts require five production rounds and six payments"
            )
        maximums = tuple(
            START_QUANTITY_MAXIMUMS[product] for product in self.products
        )
        if any(
            minimum > maximum
            for minimum, maximum in zip(self.minimum_quantities, maximums)
        ):
            raise ValueError(
                "An environment minimum quantity exceeds the random-start cap"
            )
        return maximums

    def _needs_fit(self, quantities: tuple[int, ...]) -> bool:
        maximum = self.environment.supplier.max_order_per_resource_per_round
        for resource in self.environment.supplier.resources:
            need = sum(
                quantities[index]
                * self.environment.supplier.recipes[product].get(resource, 0)
                for index, product in enumerate(self.products)
            )
            if need > maximum:
                return False
        return True

    def is_feasible(self, candidate: ContractCandidate) -> bool:
        if len(candidate.payments) != len(self.config.payment_weeks):
            return False
        if len(candidate.quantities) != len(self.products):
            return False
        if len(candidate.prices) != len(self.products):
            return False
        if any(payment < 0 for payment in candidate.payments):
            return False
        if not self.config.payment_floor <= candidate.total_payment() <= self.config.payment_cap:
            return False
        if any(
            quantity < minimum
            for quantity, minimum in zip(candidate.quantities, self.minimum_quantities)
        ):
            return False
        if not self._needs_fit(candidate.quantities):
            return False
        if any(
            not minimum <= price <= maximum
            for price, minimum, maximum in zip(
                candidate.prices, self.price_minimums, self.price_maximums
            )
        ):
            return False
        return candidate.total_payment() == candidate.face_value(
            len(self.config.production_weeks)
        )

    def evaluate(self, candidate: SearchCandidate) -> Evaluation:
        if candidate not in self.evaluation_cache:
            if not self.is_feasible(candidate):
                raise ValueError(f"Cannot evaluate infeasible candidate: {candidate}")
            result = self.solver.solve(
                candidate.to_contract(self.products, self.config),
                satisfaction_threshold=self.config.satisfaction_threshold,
            )
            self.evaluation_cache[candidate] = Evaluation.from_result(result)
        return self.evaluation_cache[candidate]

    def _utilities(self, evaluation: Evaluation) -> tuple[float, float]:
        if self.config.sweep is Sweep.MAX_CUSTOMER:
            return evaluation.supplier_utility, evaluation.customer_utility
        return evaluation.customer_utility, evaluation.supplier_utility

    def point(self, candidate: SearchCandidate, epsilon: float) -> FrontierPoint:
        evaluation = self.evaluate(candidate)
        constrained, objective = self._utilities(evaluation)
        satisfaction_met = (
            evaluation.satisfaction_probability >= self.config.minimum_satisfaction
        )
        constraint_met = satisfaction_met and constrained >= epsilon
        if not satisfaction_met:
            score = None
        elif constrained < epsilon:
            score = -INFEASIBLE_SCORE_OFFSET + constrained + 0.001 * objective
        else:
            score = objective
        return FrontierPoint(
            epsilon=epsilon,
            sweep=self.config.sweep,
            candidate=candidate,
            evaluation=evaluation,
            constraint_met=constraint_met,
            score=score,
        )

    @staticmethod
    def _score(point: FrontierPoint | None) -> float:
        return -math.inf if point is None or point.score is None else point.score

    def _payment_change_candidates(
        self,
        candidate: ContractCandidate,
        change: int,
    ) -> Iterable[ContractCandidate]:
        for index in range(len(candidate.payments)):
            payments = list(candidate.payments)
            payments[index] += change
            yield ContractCandidate(tuple(payments), candidate.quantities, candidate.prices)

    def _neighbors(
        self,
        candidate: ContractCandidate,
        *,
        quantity_deltas: tuple[int, ...],
        price_deltas: tuple[int, ...],
        payment_swaps: tuple[int, ...],
    ) -> Iterable[ContractCandidate]:
        rounds = len(self.config.production_weeks)
        seen: set[ContractCandidate] = set()

        for index, quantity in enumerate(candidate.quantities):
            for delta in quantity_deltas:
                quantities = list(candidate.quantities)
                quantities[index] = quantity + delta
                payment_change = rounds * candidate.prices[index] * delta
                provisional = ContractCandidate(
                    candidate.payments,
                    tuple(quantities),
                    candidate.prices,
                )
                for updated in self._payment_change_candidates(provisional, payment_change):
                    if updated not in seen and self.is_feasible(updated):
                        seen.add(updated)
                        yield updated

        for index, price in enumerate(candidate.prices):
            for delta in price_deltas:
                prices = list(candidate.prices)
                prices[index] = price + delta
                payment_change = rounds * candidate.quantities[index] * delta
                provisional = ContractCandidate(
                    candidate.payments,
                    candidate.quantities,
                    tuple(prices),
                )
                for updated in self._payment_change_candidates(provisional, payment_change):
                    if updated not in seen and self.is_feasible(updated):
                        seen.add(updated)
                        yield updated

        for recipient in range(len(candidate.payments)):
            for donor in range(len(candidate.payments)):
                if recipient == donor:
                    continue
                for amount in payment_swaps:
                    payments = list(candidate.payments)
                    payments[recipient] += amount
                    payments[donor] -= amount
                    updated = ContractCandidate(
                        tuple(payments), candidate.quantities, candidate.prices
                    )
                    if updated not in seen and self.is_feasible(updated):
                        seen.add(updated)
                        yield updated

    def _coordinate_ascent(
        self,
        candidate: ContractCandidate,
        epsilon: float,
        *,
        quantity_deltas: tuple[int, ...],
        price_deltas: tuple[int, ...],
        payment_swaps: tuple[int, ...],
        iterations: int,
    ) -> FrontierPoint:
        current = self.point(candidate, epsilon)
        for _iteration in range(iterations):
            best: FrontierPoint | None = None
            for neighbor in self._neighbors(
                current.candidate,
                quantity_deltas=quantity_deltas,
                price_deltas=price_deltas,
                payment_swaps=payment_swaps,
            ):
                point = self.point(neighbor, epsilon)
                if self._score(point) <= self._score(current):
                    continue
                if best is None or self._score(point) > self._score(best):
                    best = point
                if self.config.search_mode == "first":
                    break
            if best is None:
                break
            current = best
        return current

    def _optimize_start(
        self,
        candidate: ContractCandidate,
        epsilon: float,
        prune_threshold: float,
    ) -> FrontierPoint:
        coarse = self._coordinate_ascent(
            candidate,
            epsilon,
            quantity_deltas=COARSE_QUANTITY_DELTAS,
            price_deltas=COARSE_PRICE_DELTAS,
            payment_swaps=COARSE_PAYMENT_SWAPS,
            iterations=self.config.coarse_iterations,
        )
        if self._score(coarse) < prune_threshold:
            return coarse
        return self._coordinate_ascent(
            coarse.candidate,
            epsilon,
            quantity_deltas=FINE_QUANTITY_DELTAS,
            price_deltas=FINE_PRICE_DELTAS,
            payment_swaps=FINE_PAYMENT_SWAPS,
            iterations=self.config.fine_iterations,
        )

    @staticmethod
    def _sample_payments(
        total: int,
        rng: np.random.Generator,
        count: int,
    ) -> tuple[int, ...]:
        if count != 6:
            raise ValueError("F1 random starts require six payments")
        raw_payments = rng.dirichlet(np.full(count, 1.0)) * int(total)
        payments = np.round(raw_payments).astype(np.int32)
        payments[-1] = int(total) - int(payments[:-1].sum())
        return tuple(int(value) for value in payments)

    def _random_starts(
        self,
        warm_start: ContractCandidate,
        epsilon_index: int,
    ) -> tuple[ContractCandidate, ...]:
        required = self.config.number_of_starts - 1
        if required == 0:
            return (warm_start,)

        rng = np.random.default_rng(self.config.seed + epsilon_index)
        keepers: list[ContractCandidate] = []
        scored: list[tuple[float, ContractCandidate]] = []
        seen = {warm_start}
        solver_call_limit = max(
            required * 20,
            MIN_RANDOM_START_SOLVER_CALLS,
            self.config.random_start_attempts,
        )
        raw_draw_limit = max(solver_call_limit * 50, 1_000)
        solver_calls = 0
        raw_draws = 0

        while (
            len(keepers) < required
            and solver_calls < solver_call_limit
            and raw_draws < raw_draw_limit
        ):
            raw_draws += 1
            quantities = tuple(
                int(rng.integers(minimum, maximum + 1))
                for minimum, maximum in zip(
                    self.minimum_quantities, self.random_quantity_maximums
                )
            )
            prices = tuple(
                int(rng.integers(minimum, maximum + 1))
                for minimum, maximum in zip(self.price_minimums, self.price_maximums)
            )
            total = len(self.config.production_weeks) * sum(
                quantity * price for quantity, price in zip(quantities, prices)
            )
            if not self.config.payment_floor <= total <= self.config.payment_cap:
                continue

            payments = self._sample_payments(
                total,
                rng,
                len(self.config.payment_weeks),
            )
            candidate = ContractCandidate(payments, quantities, prices)
            if candidate in seen or not self.is_feasible(candidate):
                continue
            seen.add(candidate)
            solver_calls += 1
            evaluation = self.evaluate(candidate)
            scored.append((evaluation.satisfaction_probability, candidate))
            if evaluation.satisfaction_probability >= self.config.minimum_satisfaction:
                keepers.append(candidate)

        if len(keepers) < MIN_RANDOM_START_KEEPERS:
            scored.sort(
                key=lambda item: (
                    max(self.config.minimum_satisfaction - item[0], 0.0),
                    -item[0],
                )
            )
            keepers = [candidate for _p_sat, candidate in scored[:required]]
        return (warm_start, *keepers)

    def _new_checkpoint(self) -> SearchCheckpoint:
        return SearchCheckpoint(
            config_fingerprint=self.config_fingerprint,
            environment_fingerprint=self.environment_fingerprint,
            initial_candidate=self.initial_candidate,
            completed_points=[],
            warm_start=self.initial_candidate,
        )

    def _validate_checkpoint(self, checkpoint: SearchCheckpoint) -> None:
        if checkpoint.config_fingerprint != self.config_fingerprint:
            raise ValueError("Checkpoint search configuration does not match this run")
        if checkpoint.environment_fingerprint != self.environment_fingerprint:
            raise ValueError("Checkpoint environment does not match this run")
        if checkpoint.initial_candidate != self.initial_candidate:
            raise ValueError("Checkpoint initial candidate does not match this run")
        if checkpoint.epsilon_index != len(checkpoint.completed_points):
            raise ValueError("Checkpoint epsilon progress is inconsistent")
        if checkpoint.epsilon_index > len(self.config.epsilon_grid):
            raise ValueError("Checkpoint is beyond the configured epsilon grid")
        if checkpoint.start_index > len(checkpoint.current_starts):
            raise ValueError("Checkpoint start progress is inconsistent")

    def run(self, *, resume: bool = False) -> list[FrontierPoint]:
        checkpoint = load_checkpoint(self.checkpoint_path) if resume else self._new_checkpoint()
        self._validate_checkpoint(checkpoint)
        if checkpoint.status == "complete":
            return list(checkpoint.completed_points)

        while checkpoint.epsilon_index < len(self.config.epsilon_grid):
            epsilon = self.config.epsilon_grid[checkpoint.epsilon_index]
            warm_start = checkpoint.warm_start or self.initial_candidate
            if not checkpoint.current_starts:
                checkpoint.current_starts = self._random_starts(
                    warm_start, checkpoint.epsilon_index
                )
                checkpoint.start_index = 0
                checkpoint.current_best = None
                save_checkpoint_atomic(self.checkpoint_path, checkpoint)

            while checkpoint.start_index < len(checkpoint.current_starts):
                previous_best = checkpoint.current_best
                prune_threshold = (
                    self._score(checkpoint.current_best) - self.config.prune_margin
                    if checkpoint.current_best is not None
                    else -math.inf
                )
                point = self._optimize_start(
                    checkpoint.current_starts[checkpoint.start_index],
                    epsilon,
                    prune_threshold,
                )
                print(
                    f"[epsilon {checkpoint.epsilon_index + 1}/"
                    f"{len(self.config.epsilon_grid)} value={epsilon:g}] "
                    f"start {checkpoint.start_index + 1}/"
                    f"{len(checkpoint.current_starts)} "
                    f"constraint_met={point.constraint_met} score={point.score}",
                    flush=True,
                )
                if checkpoint.current_best is None or self._score(point) > self._score(
                    checkpoint.current_best
                ):
                    checkpoint.current_best = point
                checkpoint.start_index += 1
                if (
                    len(checkpoint.current_starts) == 3
                    and checkpoint.start_index == 2
                    and (
                        point.constraint_met
                        or (previous_best is not None and previous_best.constraint_met)
                    )
                ):
                    checkpoint.start_index = 3
                save_checkpoint_atomic(self.checkpoint_path, checkpoint)

            if checkpoint.current_best is None:
                raise RuntimeError("F1 search completed no starts for an epsilon point")
            checkpoint.completed_points.append(checkpoint.current_best)
            checkpoint.warm_start = checkpoint.current_best.candidate
            checkpoint.epsilon_index += 1
            checkpoint.start_index = 0
            checkpoint.current_starts = ()
            checkpoint.current_best = None
            save_checkpoint_atomic(self.checkpoint_path, checkpoint)

        checkpoint.status = "complete"
        save_checkpoint_atomic(self.checkpoint_path, checkpoint)
        return list(checkpoint.completed_points)

    def result_payload(self, points: list[FrontierPoint]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frontier": "F1",
            "status": "complete",
            "environment_id": self.environment.environment_id,
            "environment_name": self.environment.environment_name,
            "environment_axes": dict(self.environment.axes),
            "environment_fingerprint": self.environment_fingerprint,
            "config": self.config.to_dict(),
            "config_fingerprint": self.config_fingerprint,
            "initial_candidate": self.initial_candidate.to_dict(),
            "points": [point.to_dict() for point in points],
        }
