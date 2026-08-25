"""Reusable API and CLI for exact-delivery grim-trigger contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bellman import OneStepSolver, ProductionRoundInput
from contract import Contract
from environment import Environment
from runtime import TransitionRuntime


class RCCSolver:
    """Environment-bound modular solver reusable across many contracts."""

    def __init__(self, environment: Environment) -> None:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        self.environment = environment
        self.bellman_solver = OneStepSolver(environment)
        self.transition_runtime = TransitionRuntime(environment)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> RCCSolver:
        """Construct a reusable solver from an environment JSON file."""
        return cls(Environment.from_json(path))

    def solve(
        self,
        contract: Contract,
        satisfaction_threshold: float = 0.95,
        policy_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Solve one contract while retaining environment-bound runtime state."""
        if not isinstance(contract, Contract):
            raise TypeError("contract must be a Contract")

        transitions = self.transition_runtime.for_contract(contract)
        round_inputs = tuple(
            ProductionRoundInput(
                production=production,
                payment=payment,
                transitions=transitions.for_week(production.week),
            )
            for payment, production in zip(
                contract.payment_schedule[:-1],
                contract.production_schedule,
            )
        )
        if policy_path is None:
            state = self.bellman_solver.solve_rounds_from_final_payment(
                round_inputs,
                contract.payment_schedule[-1],
                satisfaction_threshold,
            )
        else:
            state, policy = (
                self.bellman_solver.solve_rounds_from_final_payment_with_policy(
                    round_inputs,
                    contract.payment_schedule[-1],
                    satisfaction_threshold,
                )
            )
            self._save_policy(policy_path, policy, satisfaction_threshold)
        state.stacked().block_until_ready()

        initial_state = (
            (0,) * len(self.environment.supplier.resources)
            + (self.environment.supplier.starting_capital,)
        )
        satisfaction_probability = float(
            state.satisfaction_probability[initial_state]
        )
        expected_supplier_utility = float(state.supplier_utility[initial_state])
        expected_customer_value = float(state.customer_value[initial_state])
        expected_total_paid = float(state.total_paid[initial_state])
        expected_customer_utility = (
            float(self.environment.customer.starting_budget)
            + (expected_customer_value - expected_total_paid)
        )
        return {
            "P_sat": satisfaction_probability,
            "E_supplier_utility": expected_supplier_utility,
            "E_customer_utility": expected_customer_utility,
        }

    def _save_policy(
        self,
        path: str | Path,
        policy: np.ndarray,
        satisfaction_threshold: float,
    ) -> None:
        """Save a grim-trigger policy in the performance loader's NPZ format."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        policy_axes = (
            ["round"]
            + [
                f"inv_{resource.lower()}"
                for resource in self.bellman_solver.resource_names
            ]
            + ["budget", "scenario", "ingredient"]
        )
        with output_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                policies=policy,
                SAT_THRESH=np.float32(satisfaction_threshold),
                PRICE_RANGES=np.asarray(
                    self.bellman_solver.price_kernel.scenarios,
                    dtype=np.float32,
                ),
                PRICE_PROB=np.asarray(
                    self.bellman_solver.price_kernel.probabilities,
                    dtype=np.float64,
                ),
                policy_axes=np.array(json.dumps(policy_axes)),
            )


def main() -> None:
    """Run one modular contract solve from the command line."""
    parser = argparse.ArgumentParser(
        description="Modular exact-delivery grim-trigger RC solver"
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--sat-thresh", type=float, default=0.95)
    parser.add_argument(
        "--save-policy",
        type=Path,
        help="Optionally save the selected grim-trigger policy as an NPZ file",
    )
    args = parser.parse_args()

    solver = RCCSolver.from_json(args.environment)
    contract = Contract.from_json(args.contract)
    result = solver.solve(
        contract,
        args.sat_thresh,
        policy_path=args.save_policy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
