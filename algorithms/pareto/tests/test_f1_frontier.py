from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PARETO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOT = PARETO_ROOT.parent / "rc-solver"
for path in (PARETO_ROOT.parent, SOLVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pareto.generation import f1_base_search
from pareto.generation import f1_frontier as f1
from contract import Contract
from environment import Environment


class FakeSolver:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, contract: Contract, satisfaction_threshold: float = 0.95):
        self.calls += 1
        total_payment = sum(week.amount for week in contract.payment_schedule)
        total_value = sum(
            quantity * {"A": 12, "B": 6, "C": 3}[product]
            for week in contract.production_schedule
            for product, quantity in week.product_quantities.items()
        )
        return {
            "P_sat": 1.0,
            "E_supplier_utility": total_payment - 100.0,
            "E_customer_utility": 200.0 + total_value - total_payment,
        }


class NoCallSolver:
    def solve(self, contract: Contract, satisfaction_threshold: float = 0.95):
        raise AssertionError("a complete checkpoint should not invoke the solver")


class SatisfactionSolver:
    def __init__(self, probability) -> None:
        self.probability = probability
        self.calls = 0

    def solve(self, contract: Contract, satisfaction_threshold: float = 0.95):
        self.calls += 1
        probability = (
            self.probability(contract)
            if callable(self.probability)
            else self.probability
        )
        return {
            "P_sat": float(probability),
            "E_supplier_utility": 0.0,
            "E_customer_utility": 0.0,
        }


class MetricsSolver:
    def __init__(
        self,
        probability: float,
        supplier_utility: float,
        customer_utility: float,
    ) -> None:
        self.probability = probability
        self.supplier_utility = supplier_utility
        self.customer_utility = customer_utility

    def solve(self, contract: Contract, satisfaction_threshold: float = 0.95):
        return {
            "P_sat": self.probability,
            "E_supplier_utility": self.supplier_utility,
            "E_customer_utility": self.customer_utility,
        }


def candidate_vector(candidate: f1.ContractCandidate) -> tuple[int, ...]:
    return candidate.payments + candidate.quantities + candidate.prices


class EnvironmentSelectionTests(unittest.TestCase):
    def test_loads_repository_environment_by_id(self) -> None:
        path, environment = f1.load_environment_selection(
            environment_id=3,
            environment_path=None,
        )

        self.assertEqual(
            path,
            (PARETO_ROOT.parents[1] / "envs" / "3" / "environment.json").resolve(),
        )
        self.assertEqual(environment.environment_id, 3)
        self.assertEqual(environment.axes["operating_conditions"], "moderate")
        self.assertEqual(
            environment.probability_dist.delivery_distribution.minimum_received,
            "floor(3*ordered/4)",
        )

    def test_keeps_explicit_environment_path_compatibility(self) -> None:
        explicit = (
            SOLVER_ROOT / "tests" / "data" / "environment_09.json"
        )
        path, environment = f1.load_environment_selection(
            environment_id=None,
            environment_path=explicit,
        )

        self.assertEqual(path, explicit.resolve())
        self.assertEqual(environment.environment_id, 9)

    def test_requires_exactly_one_environment_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            f1.resolve_environment_path(
                environment_id=None,
                environment_path=None,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            f1.resolve_environment_path(
                environment_id=1,
                environment_path=Path("environment.json"),
            )

    def test_rejects_missing_and_mislabeled_environment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment_root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "environment ID 7"):
                f1.load_environment_selection(
                    environment_id=7,
                    environment_path=None,
                    environment_root=environment_root,
                )

            payload = json.loads(
                (
                    PARETO_ROOT.parents[1]
                    / "envs"
                    / "1"
                    / "environment.json"
                ).read_text()
            )
            mislabeled = environment_root / "2" / "environment.json"
            mislabeled.parent.mkdir(parents=True)
            mislabeled.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "declares ID 1"):
                f1.load_environment_selection(
                    environment_id=2,
                    environment_path=None,
                    environment_root=environment_root,
                )


class F1FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = Environment.from_json(
            SOLVER_ROOT / "tests" / "data" / "environment_09.json"
        )
        cls.initial_contract = Contract.from_json(
            PARETO_ROOT / "data" / "f1_initial_contract.json"
        )
        cls.initial_candidate = f1.ContractCandidate.from_contract(
            cls.initial_contract, cls.environment.supplier.products
        )

    def config(self) -> f1.F1SearchConfig:
        return f1.F1SearchConfig(
            sweep=f1.Sweep.MAX_CUSTOMER,
            epsilon_grid=(0.0, 10.0),
            number_of_starts=1,
            coarse_iterations=0,
            fine_iterations=0,
            random_start_attempts=0,
        )

    def random_search(
        self,
        solver,
        **overrides,
    ) -> f1.F1FrontierSearch:
        settings = {
            "sweep": f1.Sweep.MAX_CUSTOMER,
            "epsilon_grid": (0.0,),
            "number_of_starts": 3,
            "coarse_iterations": 0,
            "fine_iterations": 0,
            "seed": 42,
        }
        settings.update(overrides)
        return f1.F1FrontierSearch(
            solver,
            self.environment,
            f1.F1SearchConfig(**settings),
            self.initial_candidate,
            Path(tempfile.gettempdir()) / "unused-f1-checkpoint.json",
            environment_fingerprint="environment-fixture",
        )

    def test_random_start_quantity_caps_match_contractbench(self) -> None:
        self.assertEqual(
            self.random_search(SatisfactionSolver(1.0)).random_quantity_maximums,
            (6, 8, 8),
        )

    def test_default_initial_contract_matches_contractbench_seed(self) -> None:
        self.assertEqual(
            f1.DEFAULT_INITIAL_CONTRACT_PATH,
            PARETO_ROOT / "data" / "f1_initial_contract.json",
        )
        self.assertEqual(
            candidate_vector(self.initial_candidate),
            (30, 30, 30, 30, 30, 30, 2, 2, 2, 10, 5, 3),
        )

    def test_base_thresholds_use_contractbench_exact_comparisons(self) -> None:
        below_satisfaction = self.random_search(
            SatisfactionSolver(0.95 - 5e-13), number_of_starts=1
        ).point(self.initial_candidate, 0.0)
        self.assertFalse(below_satisfaction.constraint_met)
        self.assertIsNone(below_satisfaction.score)

        below_epsilon = self.random_search(
            SatisfactionSolver(0.95), number_of_starts=1
        ).point(self.initial_candidate, 5e-13)
        self.assertFalse(below_epsilon.constraint_met)
        self.assertEqual(below_epsilon.score, -1_000_000.0)

        exact = self.random_search(
            SatisfactionSolver(0.95), number_of_starts=1
        ).point(self.initial_candidate, 0.0)
        self.assertTrue(exact.constraint_met)
        self.assertEqual(exact.score, 0.0)

    def test_random_starts_match_frozen_contractbench_vectors(self) -> None:
        expected = {
            0: (
                (12, 10, 24, 5, 50, 34, 3, 1, 1, 6, 6, 3),
                (7, 2, 12, 53, 24, 72, 1, 4, 2, 8, 5, 3),
            ),
            1: (
                (24, 24, 3, 0, 69, 25, 2, 3, 4, 6, 3, 2),
                (2, 4, 68, 38, 60, 13, 1, 7, 4, 8, 3, 2),
            ),
            2: (
                (68, 9, 0, 17, 14, 12, 1, 1, 3, 10, 5, 3),
                (95, 26, 4, 29, 22, 4, 1, 6, 3, 6, 4, 2),
            ),
        }
        search = self.random_search(SatisfactionSolver(1.0))

        for epsilon_index, expected_random in expected.items():
            with self.subTest(epsilon_index=epsilon_index):
                starts = search._random_starts(
                    self.initial_candidate,
                    epsilon_index,
                )
                self.assertEqual(starts[0], self.initial_candidate)
                self.assertEqual(
                    tuple(candidate_vector(candidate) for candidate in starts[1:]),
                    expected_random,
                )

    def test_payment_rounding_uses_last_payment_residual(self) -> None:
        class FixedDirichlet:
            @staticmethod
            def dirichlet(_alpha):
                return np.asarray((0.15, 0.15, 0.15, 0.15, 0.10, 0.30))

        self.assertEqual(
            f1.F1FrontierSearch._sample_payments(
                10,
                FixedDirichlet(),
                6,
            ),
            (2, 2, 2, 2, 1, 1),
        )

    def test_base_payment_swaps_follow_contractbench_order(self) -> None:
        search = self.random_search(
            NoCallSolver(),
            number_of_starts=1,
        )

        neighbors = tuple(
            search._neighbors(
                self.initial_candidate,
                quantity_deltas=(),
                price_deltas=(),
                payment_swaps=(10, 20),
            )
        )

        self.assertEqual(len(neighbors), 60)
        self.assertEqual(
            tuple(candidate.payments for candidate in neighbors[:4]),
            (
                (40, 20, 30, 30, 30, 30),
                (50, 10, 30, 30, 30, 30),
                (40, 30, 20, 30, 30, 30),
                (50, 30, 10, 30, 30, 30),
            ),
        )
        self.assertEqual(
            neighbors[10].payments,
            (20, 40, 30, 30, 30, 30),
        )

    def test_default_sorts_random_starts_by_satisfaction(self) -> None:
        solver = SatisfactionSolver(
            lambda contract: 0.90
            + (int(contract.payment_schedule[0].amount) % 11) / 100.0
        )
        search = self.random_search(solver)
        starts = search._random_starts(self.initial_candidate, 1)

        self.assertEqual(
            tuple(candidate_vector(candidate) for candidate in starts[1:]),
            (
                (40, 3, 4, 42, 56, 25, 2, 3, 3, 8, 4, 2),
                (72, 6, 5, 18, 17, 2, 1, 4, 2, 6, 3, 3),
            ),
        )

    def test_three_random_starts_keep_encounter_order(self) -> None:
        probability = lambda contract: 0.90 + (
            int(contract.payment_schedule[0].amount) % 11
        ) / 100.0
        search = self.random_search(
            SatisfactionSolver(probability),
            number_of_starts=4,
            seed=0,
        )
        starts = search._random_starts(self.initial_candidate, 0)

        np.testing.assert_allclose(
            [
                probability(
                    candidate.to_contract(search.products, search.config)
                )
                for candidate in starts[1:]
            ],
            [0.95, 0.99, 0.96],
            rtol=0.0,
            atol=1e-12,
        )

    def test_attempt_budget_counts_valid_candidates(self) -> None:
        solver = SatisfactionSolver(0.0)
        search = self.random_search(solver, random_start_attempts=0)
        starts = search._random_starts(self.initial_candidate, 0)

        self.assertEqual(solver.calls, 80)
        self.assertEqual(len(starts), 3)

    def test_raw_draw_safety_limit_precedes_payment_sampling(self) -> None:
        class MinimumRandomGenerator:
            def __init__(self) -> None:
                self.integer_calls = 0
                self.dirichlet_calls = 0

            def integers(self, low, _high):
                self.integer_calls += 1
                return low

            def dirichlet(self, _alpha):
                self.dirichlet_calls += 1
                raise AssertionError("Below-floor draws must not sample payments")

        generator = MinimumRandomGenerator()
        solver = SatisfactionSolver(0.0)
        search = self.random_search(solver)
        with patch.object(
            f1_base_search.np.random,
            "default_rng",
            return_value=generator,
        ):
            starts = search._random_starts(self.initial_candidate, 0)

        self.assertEqual(starts, (self.initial_candidate,))
        self.assertEqual(generator.integer_calls, 24_000)
        self.assertEqual(generator.dirichlet_calls, 0)
        self.assertEqual(solver.calls, 0)

    def test_candidate_requires_constant_quantities(self) -> None:
        variable = Contract.from_json(
            SOLVER_ROOT / "examples" / "variable_contract.json"
        )
        with self.assertRaisesRegex(ValueError, "same product quantities"):
            f1.ContractCandidate.from_contract(
                variable, self.environment.supplier.products
            )

    def test_third_start_is_skipped_when_either_first_start_is_feasible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            starts = (
                self.initial_candidate,
                f1.ContractCandidate(
                    (31, 29, 30, 30, 30, 30),
                    self.initial_candidate.quantities,
                    self.initial_candidate.prices,
                ),
                f1.ContractCandidate(
                    (29, 31, 30, 30, 30, 30),
                    self.initial_candidate.quantities,
                    self.initial_candidate.prices,
                ),
            )
            search = self.random_search(NoCallSolver())
            search.checkpoint_path = Path(directory) / "checkpoint.json"
            points = [
                f1.FrontierPoint(
                    0.0, f1.Sweep.MAX_CUSTOMER, starts[0],
                    f1.Evaluation(1.0, -1.0, 0.0), False, -1_000_001.0,
                ),
                f1.FrontierPoint(
                    0.0, f1.Sweep.MAX_CUSTOMER, starts[1],
                    f1.Evaluation(1.0, 1.0, 10.0), True, 10.0,
                ),
            ]
            with patch.object(search, "_random_starts", return_value=starts), patch.object(
                search, "_optimize_start", side_effect=points
            ) as optimize:
                result = search.run()

            self.assertEqual(optimize.call_count, 2)
            self.assertEqual(result, [points[1]])

    def test_third_start_runs_when_first_two_are_infeasible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            starts = (
                self.initial_candidate,
                f1.ContractCandidate(
                    (31, 29, 30, 30, 30, 30),
                    self.initial_candidate.quantities,
                    self.initial_candidate.prices,
                ),
                f1.ContractCandidate(
                    (29, 31, 30, 30, 30, 30),
                    self.initial_candidate.quantities,
                    self.initial_candidate.prices,
                ),
            )
            search = self.random_search(NoCallSolver())
            search.checkpoint_path = Path(directory) / "checkpoint.json"
            points = [
                f1.FrontierPoint(
                    0.0, f1.Sweep.MAX_CUSTOMER, starts[0],
                    f1.Evaluation(1.0, -2.0, 0.0), False, -1_000_002.0,
                ),
                f1.FrontierPoint(
                    0.0, f1.Sweep.MAX_CUSTOMER, starts[1],
                    f1.Evaluation(1.0, -1.0, 0.0), False, -1_000_001.0,
                ),
                f1.FrontierPoint(
                    0.0, f1.Sweep.MAX_CUSTOMER, starts[2],
                    f1.Evaluation(1.0, 1.0, 10.0), True, 10.0,
                ),
            ]
            with patch.object(search, "_random_starts", return_value=starts), patch.object(
                search, "_optimize_start", side_effect=points
            ) as optimize:
                result = search.run()

            self.assertEqual(optimize.call_count, 3)
            self.assertEqual(result, [points[2]])

    def test_search_checkpoints_and_resumes_without_resolving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.json"
            solver = FakeSolver()
            search = f1.F1FrontierSearch(
                solver,
                self.environment,
                self.config(),
                self.initial_candidate,
                checkpoint_path,
                environment_fingerprint="environment-fixture",
            )
            original_save = f1.save_checkpoint_atomic
            with patch.object(
                f1_base_search, "save_checkpoint_atomic", wraps=original_save
            ) as save, patch.object(
                search,
                "_random_starts",
                wraps=search._random_starts,
            ) as random_starts:
                points = search.run()

            self.assertEqual(len(points), 2)
            self.assertGreaterEqual(save.call_count, 7)
            payload = json.loads(checkpoint_path.read_text())
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["epsilon_index"], 2)
            self.assertNotIn("start_policy", payload)
            self.assertEqual(
                [call.args[1] for call in random_starts.call_args_list],
                [0, 1],
            )

            resumed = f1.F1FrontierSearch(
                NoCallSolver(),
                self.environment,
                self.config(),
                self.initial_candidate,
                checkpoint_path,
                environment_fingerprint="environment-fixture",
            ).run(resume=True)
            self.assertEqual(resumed, points)

    def test_resume_rejects_a_different_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.json"
            search = f1.F1FrontierSearch(
                FakeSolver(),
                self.environment,
                self.config(),
                self.initial_candidate,
                checkpoint_path,
                environment_fingerprint="first",
            )
            search.run()
            mismatched = f1.F1FrontierSearch(
                NoCallSolver(),
                self.environment,
                self.config(),
                self.initial_candidate,
                checkpoint_path,
                environment_fingerprint="second",
            )
            with self.assertRaisesRegex(ValueError, "environment does not match"):
                mismatched.run(resume=True)

class F1FrontierRefinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = Environment.from_json(
            SOLVER_ROOT / "tests" / "data" / "environment_09.json"
        )
        cls.anchor = f1.RefinedContractCandidate(
            payments=(35, 35, 30, 30, 30, 30),
            quantity_schedule=((3, 1, 1),) * 5,
            prices=(10.0, 5.0, 3.0),
        )

    def config(
        self,
        *,
        iterations: int = 0,
        search_mode: str = "best",
    ) -> f1.F1RefinedSearchConfig:
        return f1.F1RefinedSearchConfig(
            sweep=f1.Sweep.MAX_CUSTOMER,
            epsilon_grid=(0.0, 10.0),
            number_of_starts=1,
            refined_iterations=iterations,
            search_mode=search_mode,
        )

    def search(
        self,
        checkpoint: Path,
        *,
        solver=None,
        anchor: f1.RefinedContractCandidate | None = None,
        iterations: int = 0,
        search_mode: str = "best",
    ) -> f1.F1FrontierSearchRefined:
        return f1.F1FrontierSearchRefined(
            solver or FakeSolver(),
            self.environment,
            self.config(iterations=iterations, search_mode=search_mode),
            anchor or self.anchor,
            checkpoint,
            environment_fingerprint="environment-fixture",
        )

    def test_refined_search_inherits_base_search(self) -> None:
        self.assertTrue(issubclass(f1.F1FrontierSearchRefined, f1.F1FrontierSearch))

    def test_refined_candidate_preserves_schedule_and_fractional_price(self) -> None:
        contract = Contract.from_mapping(
            {
                "product_prices": {"A": 8.4375, "B": 8, "C": 5},
                "production_schedule": [
                    {
                        "week": week,
                        "product_quantities": {
                            "A": 4 if week == 10 else 3,
                            "B": 1,
                            "C": 1,
                        },
                    }
                    for week in (2, 4, 6, 8, 10)
                ],
                "payment_schedule": [
                    {"week": week, "amount": amount}
                    for week, amount in zip(
                        (1, 3, 5, 7, 9, 11), (40, 35, 35, 35, 35, 20)
                    )
                ],
            }
        )
        candidate = f1.RefinedContractCandidate.from_contract(
            contract, self.environment.supplier.products
        )
        self.assertEqual(candidate.quantity_schedule[-1], (4, 1, 1))
        self.assertEqual(candidate.prices[0], 8.4375)

    def test_round_redistribution_and_local_box_match_contractbench(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = self.search(Path(directory) / "checkpoint.json")
            neighbors = dict(search._refined_neighbors(self.anchor))
            redistributed = neighbors["move_A_r1_to_r2"]
            self.assertEqual(redistributed.quantity_schedule[0], (2, 1, 1))
            self.assertEqual(redistributed.quantity_schedule[1], (4, 1, 1))
            self.assertEqual(redistributed.total_payment(), self.anchor.total_payment())

            outside_schedule = [list(row) for row in self.anchor.quantity_schedule]
            outside_schedule[0][0] += 2
            outside = f1.RefinedContractCandidate(
                payments=self.anchor.payments,
                quantity_schedule=tuple(tuple(row) for row in outside_schedule),
                prices=self.anchor.prices,
            )
            self.assertFalse(search.is_feasible(outside))

    def test_refined_validity_does_not_apply_base_price_bounds(self) -> None:
        anchor = f1.RefinedContractCandidate(
            payments=(20, 20, 15, 15, 15, 15),
            quantity_schedule=((1, 1, 1),) * 5,
            prices=(13.0, 3.0, 4.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            search = self.search(Path(directory) / "checkpoint.json", anchor=anchor)
            self.assertTrue(search.is_feasible(anchor))
            self.assertGreater(
                anchor.prices[0], self.environment.customer.private_values["A"]
            )

    def test_refined_initialization_does_not_compute_base_price_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            f1.F1FrontierSearch,
            "_price_minimums",
            side_effect=AssertionError("refined search computed base price bounds"),
        ):
            search = self.search(Path(directory) / "checkpoint.json")

        self.assertEqual(search.price_minimums, ())
        self.assertEqual(search.price_maximums, ())
        self.assertTrue(search.is_feasible(self.anchor))

    def test_refined_and_base_share_contractbench_scoring(self) -> None:
        base_candidate = f1.ContractCandidate(
            payments=self.anchor.payments,
            quantities=self.anchor.quantity_schedule[0],
            prices=tuple(int(price) for price in self.anchor.prices),
        )
        cases = (
            (0.95 - 5e-13, 10.0, False, None),
            (
                0.95,
                10.0 + 5e-13,
                False,
                -1_000_000.0 + 10.0 + 0.001 * 30.0,
            ),
            (0.95, 10.0, True, 30.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (probability, epsilon, constraint_met, score) in enumerate(cases):
                with self.subTest(probability=probability, epsilon=epsilon):
                    base = f1.F1FrontierSearch(
                        MetricsSolver(probability, 10.0, 30.0),
                        self.environment,
                        f1.F1SearchConfig(
                            sweep=f1.Sweep.MAX_CUSTOMER,
                            epsilon_grid=(epsilon,),
                            number_of_starts=1,
                        ),
                        base_candidate,
                        root / f"base-{index}.json",
                        environment_fingerprint="environment-fixture",
                    ).point(base_candidate, epsilon)
                    refined = self.search(
                        root / f"refined-{index}.json",
                        solver=MetricsSolver(probability, 10.0, 30.0),
                    ).point(self.anchor, epsilon)

                    self.assertEqual(refined.constraint_met, constraint_met)
                    self.assertEqual(refined.score, score)
                    self.assertEqual(
                        (refined.constraint_met, refined.score),
                        (base.constraint_met, base.score),
                    )
                    if score is None:
                        self.assertEqual(
                            f1.F1FrontierSearch._score(refined), -float("inf")
                        )

    def test_shared_scoring_preserves_max_supplier_orientation(self) -> None:
        epsilon = 30.0
        config = f1.F1RefinedSearchConfig(
            sweep=f1.Sweep.MAX_SUPPLIER,
            epsilon_grid=(epsilon,),
            number_of_starts=1,
            refined_iterations=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            search = f1.F1FrontierSearchRefined(
                MetricsSolver(0.95, 10.0, 30.0),
                self.environment,
                config,
                self.anchor,
                Path(directory) / "checkpoint.json",
                environment_fingerprint="environment-fixture",
            )
            point = search.point(self.anchor, epsilon)

        self.assertTrue(point.constraint_met)
        self.assertEqual(point.score, 10.0)

    def test_refined_first_improvement_uses_contractbench_neighbor_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            search = self.search(
                Path(directory) / "checkpoint.json",
                iterations=1,
                search_mode="first",
            )
            point = search._local_ascent(self.anchor, epsilon=0.0)
            self.assertEqual(point.trace[0]["move"], "qty_A_r1+1")
            self.assertEqual(point.candidate.quantity_schedule[0], (4, 1, 1))

    def test_refined_checkpoint_round_trip_resumes_without_solving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            points = self.search(checkpoint).run()
            self.assertEqual(len(points), 2)
            self.assertIsInstance(points[0].candidate, f1.RefinedContractCandidate)

            resumed = self.search(checkpoint, solver=NoCallSolver()).run(resume=True)
            self.assertEqual(resumed, points)


class F1FrontierCliTests(unittest.TestCase):
    def test_refined_rejects_explicit_base_only_options_before_loading(self) -> None:
        options = (
            ("--starts", "3"),
            ("--coarse-iterations", "10"),
            ("--fine-iterations", "50"),
            ("--prune-margin", "20"),
            ("--random-start-attempts", "80"),
            ("--seed", "42"),
        )
        required = [
            "f1_frontier.py",
            "--env",
            "1",
            "--initial-contract",
            str(f1.DEFAULT_INITIAL_CONTRACT_PATH),
            "--sweep",
            "max-customer",
            "--epsilon-grid",
            "0",
            "--refined",
            "--checkpoint",
            "checkpoint.json",
            "--output",
            "output.json",
        ]

        for option, value in options:
            with self.subTest(option=option):
                stderr = io.StringIO()
                with patch.object(sys, "argv", [*required, option, value]), patch.object(
                    sys, "stderr", stderr
                ), patch.object(
                    f1,
                    "load_environment_selection",
                    side_effect=AssertionError("CLI loaded an environment"),
                ) as load_environment:
                    with self.assertRaises(SystemExit) as raised:
                        f1.main()

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(option, stderr.getvalue())
                load_environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
