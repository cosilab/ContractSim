from __future__ import annotations

import unittest
from pathlib import Path

from analysis.performance.scripts.analyze_campaign import (
    _add_regret_metrics,
    _aggregate_by_role,
    _default_output_dir,
)


class CampaignRegretTests(unittest.TestCase):
    def test_regret_uses_matching_focal_rcc_run(self) -> None:
        rows = [
            {
                "environment_id": 1,
                "contract_id": "contract-1",
                "seed": 42,
                "customer": "rc",
                "supplier": "re",
                "valid": True,
                "customer_utility": 8.0,
                "supplier_utility": 3.0,
                "customer_compliant_utility_zeroed": 7.0,
                "supplier_compliant_utility_zeroed": 2.0,
            },
            {
                "environment_id": 1,
                "contract_id": "contract-1",
                "seed": 42,
                "customer": "rcc",
                "supplier": "re",
                "valid": True,
                "customer_utility": 10.0,
                "supplier_utility": 4.0,
                "customer_compliant_utility_zeroed": 9.0,
                "supplier_compliant_utility_zeroed": 4.0,
            },
            {
                "environment_id": 1,
                "contract_id": "contract-1",
                "seed": 42,
                "customer": "rc",
                "supplier": "rcc",
                "valid": True,
                "customer_utility": 9.0,
                "supplier_utility": 7.0,
                "customer_compliant_utility_zeroed": 8.0,
                "supplier_compliant_utility_zeroed": 6.0,
            },
        ]

        _add_regret_metrics(rows)

        self.assertEqual(rows[0]["customer_ex_post_regret"], 2.0)
        self.assertEqual(
            rows[0]["customer_ex_post_compliant_regret_zeroed"], 2.0
        )
        self.assertEqual(rows[0]["supplier_ex_post_regret"], 4.0)
        self.assertEqual(
            rows[0]["supplier_ex_post_compliant_regret_zeroed"], 4.0
        )
        self.assertEqual(rows[1]["customer_ex_post_regret"], 0.0)
        self.assertIsNone(rows[1]["supplier_ex_post_regret"])

    def test_role_aggregates_separate_focal_and_counterparty_metrics(self) -> None:
        rows = [
            {
                "environment_id": 1,
                "contract_id": "contract-1",
                "seed": 1,
                "customer": "rc",
                "supplier": "rcc",
                "customer_utility": 10.0,
                "supplier_utility": 20.0,
                "delivery_shortfalls": 2.0,
            },
            {
                "environment_id": 1,
                "contract_id": "contract-1",
                "seed": 2,
                "customer": "rc",
                "supplier": "rcc",
                "customer_utility": 14.0,
                "supplier_utility": 24.0,
                "delivery_shortfalls": 4.0,
            },
            {
                "environment_id": 2,
                "contract_id": "contract-2",
                "seed": 1,
                "customer": "rc",
                "supplier": "rcc",
                "customer_utility": 20.0,
                "supplier_utility": 30.0,
                "delivery_shortfalls": 8.0,
            },
        ]

        result = _aggregate_by_role(
            rows,
            ["customer_utility", "supplier_utility", "delivery_shortfalls"],
        )

        customer = result["customer"]["focal-rc_counterparty-rcc"]
        supplier = result["supplier"]["focal-rcc_counterparty-rc"]
        self.assertEqual(
            customer,
            {
                "runs": 3,
                "cases": 2,
                "customer_utility": 16.0,
                "customer_utility_std": 5.656854,
            },
        )
        self.assertEqual(
            supplier,
            {
                "runs": 3,
                "cases": 2,
                "supplier_utility": 26.0,
                "supplier_utility_std": 5.656854,
                "delivery_shortfalls": 5.5,
                "delivery_shortfalls_std": 3.535534,
            },
        )


class InferenceOutputTests(unittest.TestCase):
    def test_routes_main_and_ablation_campaigns_to_separate_arms(self) -> None:
        main = Path("/repo/results/performance-main/main-campaign")
        ablation = Path("/repo/results/performance-abl/abl-campaign")

        self.assertEqual(
            _default_output_dir(main).parts[-2:],
            ("main", "main-campaign"),
        )
        self.assertEqual(
            _default_output_dir(ablation).parts[-2:],
            ("abl", "abl-campaign"),
        )

    def test_unknown_campaign_parent_requires_explicit_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass --output-dir"):
            _default_output_dir(Path("/tmp/custom/campaign"))


if __name__ == "__main__":
    unittest.main()
