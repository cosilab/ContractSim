from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PARETO_ROOT = Path(__file__).resolve().parents[1]
if str(PARETO_ROOT) not in sys.path:
    sys.path.insert(0, str(PARETO_ROOT))

from plot_f1_frontier import load_f1_file, pareto_envelope


def point(
    supplier: float,
    customer: float,
    *,
    feasible: bool = True,
    p_sat: float = 0.96,
) -> dict:
    return {
        "epsilon": supplier,
        "sweep": "max-customer",
        "P_sat": p_sat,
        "E_supplier_utility": supplier,
        "E_customer_utility": customer,
        "constraint_met": feasible,
        "candidate": {},
    }


class PlotF1FrontierTests(unittest.TestCase):
    def test_envelope_removes_dominated_duplicate_and_low_satisfaction_points(self) -> None:
        frontier = pareto_envelope(
            [
                point(10, 100),
                point(20, 90),
                point(15, 80),
                point(20, 90),
                point(30, 120, p_sat=0.94),
            ]
        )
        self.assertEqual(
            [
                (row["E_supplier_utility"], row["E_customer_utility"])
                for row in frontier
            ],
            [(10.0, 100.0), (20.0, 90.0)],
        )

    def test_envelope_retains_global_candidate_that_missed_its_epsilon(self) -> None:
        frontier = pareto_envelope(
            [
                point(10, 100),
                point(30, 120, feasible=False),
            ]
        )

        self.assertEqual(len(frontier), 1)
        self.assertEqual(
            (
                frontier[0]["E_supplier_utility"],
                frontier[0]["E_customer_utility"],
                frontier[0]["constraint_met"],
            ),
            (30.0, 120.0, False),
        )

    def test_global_satisfaction_floor_uses_contractbench_tolerance(self) -> None:
        for probability, retained in (
            (0.95, True),
            (0.95 - 5e-13, True),
            (0.95 - 2e-12, False),
        ):
            with self.subTest(probability=probability):
                self.assertEqual(
                    bool(pareto_envelope([point(10, 100, p_sat=probability)])),
                    retained,
                )

    def test_file_loader_skips_malformed_numeric_rows(self) -> None:
        malformed_utility = point(20, 90)
        malformed_utility["E_supplier_utility"] = "not-a-number"
        malformed_probability = point(30, 80)
        malformed_probability["P_sat"] = None

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.json"
            path.write_text(
                json.dumps(
                    [
                        point(10, 100),
                        malformed_utility,
                        malformed_probability,
                    ]
                )
            )
            _metadata, candidates = load_f1_file(path)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            (
                candidates[0]["E_supplier_utility"],
                candidates[0]["E_customer_utility"],
            ),
            (10.0, 100.0),
        )


if __name__ == "__main__":
    unittest.main()
