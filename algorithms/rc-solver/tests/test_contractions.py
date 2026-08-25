from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from bellman.contractions import sequential_contraction_steps


class ContractionEquationTests(unittest.TestCase):
    def test_steps_match_contractbench_resource_order(self) -> None:
        self.assertEqual(
            sequential_contraction_steps(3),
            (
                ("cfi", "aijkb", "acjkfb"),
                ("dgj", "acjkfb", "acdkfgb"),
                ("ehk", "acdkfgb", "acdefghb"),
            ),
        )

    def test_rejects_invalid_resource_counts(self) -> None:
        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError)):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    sequential_contraction_steps(value)


if __name__ == "__main__":
    unittest.main()
