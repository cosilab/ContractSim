from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PARETO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOT = PARETO_ROOT.parent / "rc-solver"
for path in (PARETO_ROOT, PARETO_ROOT.parent, SOLVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pareto.generation.f1_common import Sweep
from pareto.generation.f1_refined_search_prereq import dominates, prepare_environment
from pareto.generation.run_f1_refined_anchors import (
    anchor_epsilon_grid,
    ordered_anchors,
)


def _point(supplier: float, customer: float, p_sat: float = 1.0) -> dict:
    return {
        "E_supplier_utility": supplier,
        "E_customer_utility": customer,
        "P_sat": p_sat,
    }


class RefinedAnchorPrerequisiteTests(unittest.TestCase):
    def test_dominance_requires_one_strict_improvement(self) -> None:
        candidate = _point(10.0, 20.0)

        self.assertFalse(dominates(_point(10.0, 20.0), candidate))
        self.assertTrue(dominates(_point(11.0, 20.0), candidate))
        self.assertTrue(dominates(_point(10.0, 21.0), candidate))
        self.assertFalse(dominates(_point(11.0, 19.0), candidate))

    def test_prepares_only_eligible_nondominated_accepted_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_qual = root / "contract-qual"
            results = root / "pareto-search"
            solves = contract_qual / "solves" / "env-03"
            frontier_dir = results / "env-03" / "f1-merged"
            solves.mkdir(parents=True)
            frontier_dir.mkdir(parents=True)

            jobs = [
                self._job("kept", accepted=True),
                self._job("dominated", accepted=True),
                self._job("low-sat", accepted=True),
                self._job("intermediate", accepted=False),
            ]
            (contract_qual / "jobs.jsonl").write_text(
                "".join(json.dumps(job) + "\n" for job in jobs)
            )
            self._write_solve(solves, "kept", 5.0, 5.0, 0.96)
            self._write_solve(solves, "dominated", 3.0, 3.0, 0.99)
            self._write_solve(solves, "low-sat", 8.0, 8.0, 0.90)
            self._write_solve(solves, "intermediate", 9.0, 9.0, 1.0)
            (frontier_dir / "f1_frontier.json").write_text(
                json.dumps(
                    {
                        "environment_id": 3,
                        "environment_fingerprint": "digest",
                        "points": [_point(4.0, 4.0)],
                    }
                )
            )

            payload = prepare_environment(
                3,
                contract_qual_root=contract_qual,
                results_root=results,
            )

        self.assertEqual(payload["anchor_count"], 1)
        self.assertEqual(payload["anchors"][0]["anchor_id"], "kept")
        self.assertEqual(payload["selection_counts"]["accepted_jobs"], 3)
        self.assertEqual(payload["selection_counts"]["dominated"], 1)
        self.assertEqual(
            payload["selection_counts"]["below_satisfaction_floor"], 1
        )

    @staticmethod
    def _job(solve_hash: str, *, accepted: bool) -> dict:
        return {
            "environment_id": 3,
            "solve_hash": solve_hash,
            "accepted_final_record_ids": [f"record/{solve_hash}"] if accepted else [],
            "record_ids": [f"record/{solve_hash}"],
            "contingency": {"combination": "grim_trigger"},
            "contract": {
                "product_prices": {"A": 1, "B": 1, "C": 1},
                "production_schedule": [],
                "payment_schedule": [],
            },
        }

    @staticmethod
    def _write_solve(
        directory: Path,
        solve_hash: str,
        supplier: float,
        customer: float,
        p_sat: float,
    ) -> None:
        (directory / f"{solve_hash}.json").write_text(
            json.dumps(
                {
                    "solve_hash": solve_hash,
                    "solver_status": "solved",
                    "E_caterer_utility": supplier,
                    "E_customer_utility": customer,
                    "P_sat": p_sat,
                }
            )
        )


class RefinedAnchorRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchors = [
            {
                "anchor_id": "middle",
                "E_supplier_utility": 20.0,
                "E_customer_utility": 50.0,
            },
            {
                "anchor_id": "low-customer",
                "E_supplier_utility": 30.0,
                "E_customer_utility": 40.0,
            },
            {
                "anchor_id": "low-supplier",
                "E_supplier_utility": 10.0,
                "E_customer_utility": 60.0,
            },
        ]

    def test_orders_by_the_selected_focus_role(self) -> None:
        customer_order = ordered_anchors(self.anchors, Sweep.MAX_CUSTOMER)
        supplier_order = ordered_anchors(self.anchors, Sweep.MAX_SUPPLIER)

        self.assertEqual(
            [anchor["anchor_id"] for anchor in customer_order],
            ["low-customer", "middle", "low-supplier"],
        )
        self.assertEqual(
            [anchor["anchor_id"] for anchor in supplier_order],
            ["low-supplier", "middle", "low-customer"],
        )

    def test_epsilon_grid_centers_the_constrained_role(self) -> None:
        anchor = self.anchors[0]

        self.assertEqual(
            anchor_epsilon_grid(anchor, Sweep.MAX_CUSTOMER, 2.0),
            (18.0, 20.0, 22.0),
        )
        self.assertEqual(
            anchor_epsilon_grid(anchor, Sweep.MAX_SUPPLIER, 2.0),
            (48.0, 50.0, 52.0),
        )


if __name__ == "__main__":
    unittest.main()
