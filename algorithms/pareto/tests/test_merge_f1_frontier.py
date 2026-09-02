from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ALGORITHMS_ROOT = Path(__file__).resolve().parents[2]
if str(ALGORITHMS_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_ROOT))

from pareto.generation.f1_merge_base_refinement import (  # noqa: E402
    build_frontier_payload,
    default_sources,
    write_frontier_outputs,
)


def _point(supplier: float, customer: float, *, p_sat: float = 0.96) -> dict:
    return {
        "epsilon": supplier,
        "sweep": "max-customer",
        "P_sat": p_sat,
        "E_supplier_utility": supplier,
        "E_customer_utility": customer,
        "constraint_met": True,
        "candidate": {},
    }


class MergeF1FrontierTests(unittest.TestCase):
    def test_merges_base_and_refined_role_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            paths = default_sources(data_root, 3)
            points = (
                _point(10, 100),
                _point(20, 90),
                _point(15, 80),
                _point(30, 120, p_sat=0.94),
            )
            for path, point in zip(paths, points):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "environment_id": 3,
                            "environment_name": "test",
                            "environment_fingerprint": "same",
                            "points": [point],
                        }
                    )
                )

            payload = build_frontier_payload(paths)
            output_directory = data_root / "env-03" / "f1-merged"
            write_frontier_outputs(output_directory, payload)

            self.assertEqual(payload["candidate_count"], 4)
            self.assertEqual(payload["frontier_point_count"], 2)
            self.assertEqual(
                [
                    (row["E_supplier_utility"], row["E_customer_utility"])
                    for row in payload["points"]
                ],
                [(10.0, 100.0), (20.0, 90.0)],
            )
            self.assertTrue((output_directory / "f1_frontier.json").is_file())
            self.assertTrue((output_directory / "f1_frontier.csv").is_file())


if __name__ == "__main__":
    unittest.main()
