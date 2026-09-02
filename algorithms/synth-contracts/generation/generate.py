#!/usr/bin/env python3
"""Search for welfare-maximizing grim contracts in fixed P_sat bands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pareto.generation.f1_base_search import F1FrontierSearch
from pareto.generation.f1_common import (
    DEFAULT_ENVIRONMENT_ROOT,
    DEFAULT_INITIAL_CONTRACT_PATH,
    Contract,
    ContractCandidate,
    Evaluation,
    F1SearchConfig,
    FrontierPoint,
    MIN_RANDOM_START_SOLVER_CALLS,
    Sweep,
    file_fingerprint,
    load_environment_selection,
    save_json_atomic,
)

SATISFACTION_THRESHOLD = 0.95
P_SAT_BANDS = {
    50: (0.45, 0.55, False),
    60: (0.55, 0.65, False),
    70: (0.65, 0.75, False),
    80: (0.75, 0.85, False),
    90: (0.85, 0.95, True),
}


def in_band(probability: float, target: int) -> bool:
    """Return whether a probability belongs to one canonical target band."""
    lower, upper, include_upper = P_SAT_BANDS[target]
    if include_upper:
        return lower <= probability <= upper
    return lower <= probability < upper


def distance_to_band(probability: float, target: int) -> float:
    """Return the shortest distance from a probability to a target band."""
    lower, upper, _include_upper = P_SAT_BANDS[target]
    return max(lower - probability, probability - upper, 0.0)


class SyntheticContractSearch(F1FrontierSearch):
    """Reuse F1 mechanics with synthetic-contract constraints and objective."""

    def __init__(self, *args, target: int, **kwargs) -> None:
        if target not in P_SAT_BANDS:
            raise ValueError(f"Unknown P_sat target: {target}")
        config = args[2]
        if config.satisfaction_threshold != SATISFACTION_THRESHOLD:
            raise ValueError("Synthetic contracts must use satisfaction_threshold=0.95")
        self.target = target
        super().__init__(*args, **kwargs)

    def _needs_fit(self, quantities: tuple[int, ...]) -> bool:
        capacity = min(
            self.environment.supplier.max_order_per_resource_per_round,
            self.environment.supplier.max_inventory_per_resource,
        )
        for resource in self.environment.supplier.resources:
            need = sum(
                quantities[index]
                * self.environment.supplier.recipes[product].get(resource, 0)
                for index, product in enumerate(self.products)
            )
            if need > capacity:
                return False
        return True

    def _random_starts(
        self,
        warm_start: ContractCandidate,
        epsilon_index: int,
    ) -> tuple[ContractCandidate, ...]:
        required = self.config.number_of_starts - 1
        if required == 0:
            return (warm_start,)

        rng = np.random.default_rng(self.config.seed + epsilon_index)
        seen = {warm_start}
        scored: list[FrontierPoint] = []
        solver_call_limit = max(
            required * 20,
            MIN_RANDOM_START_SOLVER_CALLS,
            self.config.random_start_attempts,
        )
        raw_draw_limit = max(solver_call_limit * 50, 1_000)
        solver_calls = 0
        raw_draws = 0

        while solver_calls < solver_call_limit and raw_draws < raw_draw_limit:
            raw_draws += 1
            quantities = tuple(
                int(rng.integers(minimum, maximum + 1))
                for minimum, maximum in zip(
                    self.minimum_quantities, self.random_quantity_maximums
                )
            )
            prices = tuple(
                int(rng.integers(minimum, maximum + 1))
                for minimum, maximum in zip(
                    self.price_minimums, self.price_maximums
                )
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
            scored.append(self.point(candidate, float(self.target)))

        scored.sort(key=self._score, reverse=True)
        return (warm_start, *(point.candidate for point in scored[:required]))

    def point(self, candidate: ContractCandidate, epsilon: float) -> FrontierPoint:
        evaluation = self.evaluate(candidate)
        welfare = evaluation.supplier_utility + evaluation.customer_utility
        supplier_shortfall = max(0.0, -evaluation.supplier_utility)
        customer_shortfall = max(
            0.0,
            self.environment.customer.starting_budget
            - evaluation.customer_utility,
        )
        mutual = supplier_shortfall == 0.0 and customer_shortfall == 0.0
        band_met = in_band(evaluation.satisfaction_probability, self.target)
        band_distance = distance_to_band(
            evaluation.satisfaction_probability,
            self.target,
        )

        if mutual and band_met:
            score = welfare
        elif mutual:
            score = (
                -1_000_000.0
                - 1_000.0 * band_distance
                + 0.001 * welfare
            )
        else:
            score = (
                -2_000_000.0
                - 1_000.0 * (supplier_shortfall + customer_shortfall)
                - band_distance
                + 0.000001 * welfare
            )

        return FrontierPoint(
            epsilon=epsilon,
            sweep=self.config.sweep,
            candidate=candidate,
            evaluation=evaluation,
            constraint_met=mutual and band_met,
            score=score,
        )

    def result(self, point: FrontierPoint) -> dict:
        evaluation = point.evaluation
        mutual = (
            evaluation.supplier_utility >= 0.0
            and evaluation.customer_utility
            >= self.environment.customer.starting_budget
        )
        band_met = in_band(evaluation.satisfaction_probability, self.target)
        lower, upper, include_upper = P_SAT_BANDS[self.target]
        status = (
            "accepted"
            if mutual and band_met
            else "provisional"
            if mutual
            else "failed"
        )
        return {
            "status": status,
            "target_p_sat": self.target / 100.0,
            "p_sat_band": {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": include_upper,
            },
            "satisfaction_threshold": SATISFACTION_THRESHOLD,
            "mutual_benefit_floors": {
                "supplier": 0.0,
                "customer": self.environment.customer.starting_budget,
            },
            "band_met": band_met,
            "mutual_benefit_met": mutual,
            "welfare": evaluation.supplier_utility + evaluation.customer_utility,
            "candidate": point.candidate.to_dict(),
            **evaluation.to_dict(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-contract",
        type=Path,
        default=DEFAULT_INITIAL_CONTRACT_PATH,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--coarse-iterations", type=int, default=10)
    parser.add_argument("--fine-iterations", type=int, default=50)
    parser.add_argument("--random-start-attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    environment_path, environment = load_environment_selection(
        environment_id=args.env,
        environment_path=None,
        environment_root=DEFAULT_ENVIRONMENT_ROOT,
    )
    initial = ContractCandidate.from_contract(
        Contract.from_json(args.initial_contract),
        environment.supplier.products,
    )

    from rc_solver import RCCSolver

    solver = RCCSolver(environment)
    shared_cache: dict[ContractCandidate, Evaluation] = {}
    environment_fingerprint = ":".join(
        (file_fingerprint(environment_path), file_fingerprint(Path(__file__)))
    )
    results = []

    for target in P_SAT_BANDS:
        band_dir = args.out_dir / f"env-{args.env:02d}" / f"p{target:03d}"
        checkpoint = band_dir / "checkpoint.json"
        config = F1SearchConfig(
            sweep=Sweep.MAX_CUSTOMER,
            epsilon_grid=(float(target),),
            satisfaction_threshold=SATISFACTION_THRESHOLD,
            minimum_satisfaction=0.0,
            number_of_starts=args.starts,
            coarse_iterations=args.coarse_iterations,
            fine_iterations=args.fine_iterations,
            random_start_attempts=args.random_start_attempts,
            seed=args.seed,
            payment_cap=int(environment.customer.starting_budget),
        )
        search = SyntheticContractSearch(
            solver,
            environment,
            config,
            initial,
            checkpoint,
            environment_fingerprint=environment_fingerprint,
            target=target,
        )
        search.evaluation_cache = shared_cache
        point = search.run(resume=args.resume and checkpoint.exists())[0]
        result = search.result(point)
        save_json_atomic(band_dir / "result.json", result)
        contract = point.candidate.to_contract(
            environment.supplier.products,
            config,
        )
        save_json_atomic(
            band_dir / "terms.json",
            {
                "dish_prices": dict(
                    zip(
                        environment.supplier.products,
                        point.candidate.prices,
                    )
                ),
                "production_schedule": [
                    {
                        "week": week,
                        **dict(
                            zip(
                                environment.supplier.products,
                                point.candidate.quantities,
                            )
                        ),
                    }
                    for week in config.production_weeks
                ],
                "payment_schedule": [
                    {"week": week, "amount": amount}
                    for week, amount in zip(
                        config.payment_weeks,
                        point.candidate.payments,
                    )
                ],
            },
        )
        save_json_atomic(
            band_dir / "config.json",
            {
                "environment_id": args.env,
                "contingencies": ["grim_trigger"],
                "contingency_mode": "grim_trigger",
                "contingency_params": {},
            },
        )
        solver.solve(
            contract,
            satisfaction_threshold=SATISFACTION_THRESHOLD,
            policy_path=band_dir / "baselines" / "rcc" / "policy.npz",
        )
        results.append(result)

    save_json_atomic(
        args.out_dir / f"env-{args.env:02d}" / "results.json",
        {"environment_id": args.env, "results": results},
    )


if __name__ == "__main__":
    main()
