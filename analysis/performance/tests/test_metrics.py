from __future__ import annotations

import unittest

from analysis.performance.metrics import (
    analyze_run,
    compute_behavioral_metrics,
    compute_compliance,
)


def _summary(
    *,
    payments: list[dict],
    production: list[dict],
) -> dict:
    return {
        "environment_id": 1,
        "payment_history": payments,
        "production_history": production,
        "customer_remaining_budget": 100,
        "supplier_final_balance": 40,
        "delivered": {"A": 1, "B": 2, "C": 1},
    }


class PerformanceMetricsTests(unittest.TestCase):
    def test_payment_deduction_uses_nested_ratcon_schema(self) -> None:
        terms = {
            "dish_prices": {"A": 12, "B": 7, "C": 5},
            "payment_schedule": [
                {"week": 1, "amount": 50},
                {"week": 3, "amount": 50},
            ],
            "production_schedule": [{"week": 2, "A": 1, "B": 2, "C": 1}],
        }
        config = {
            "contingencies": ["payment_deduction"],
            "contingency_params": {
                "payment_deduction": {
                    "deduction_prices": {"A": 12, "B": 7, "C": 5}
                }
            },
        }
        summary = _summary(
            payments=[{"week": 1, "amount": 50}, {"week": 3, "amount": 45}],
            production=[
                {
                    "week": 2,
                    "produced": {"A": 1, "B": 2, "C": 0},
                    "purchase_cost": 20,
                }
            ],
        )

        compliance = compute_compliance(summary, terms, config)

        self.assertTrue(compliance["supplier"]["per_week"][0]["compliant"])
        self.assertTrue(compliance["customer"]["per_week"][1]["compliant"])
        self.assertEqual(
            compliance["customer"]["per_week"][1]["deduction_applied"], 5.0
        )
    def test_substitution_above_minimums_is_compliant(self) -> None:
        terms = {
            "dish_prices": {"A": 12, "B": 7, "C": 5},
            "payment_schedule": [],
            "production_schedule": [{"week": 2, "A": 2, "B": 2, "C": 1}],
        }
        config = {
            "contingencies": ["substitution"],
            "contingency_params": {
                "substitution": {"min_qty": {"A": 1, "B": 2, "C": 1}}
            },
        }
        summary = _summary(
            payments=[],
            production=[
                {
                    "week": 2,
                    "produced": {"A": 1, "B": 2, "C": 5},
                    "purchase_cost": 20,
                }
            ],
        )

        result = compute_compliance(summary, terms, config)["supplier"]["per_week"][0]

        self.assertTrue(result["compliant"])
        self.assertEqual(result["shortfall"], {"A": 0, "B": 0, "C": 0})

    def test_rollover_failure_keeps_later_payments_evaluable(self) -> None:
        terms = {
            "payment_schedule": [
                {"week": 1, "amount": 10},
                {"week": 3, "amount": 20},
            ],
            "production_schedule": [{"week": 2, "A": 3, "B": 0, "C": 0}],
        }
        config = {
            "contingencies": ["rollover"],
            "contingency_params": {"rollover": {"max_deficit": 2}},
        }
        production = [
            {
                "week": 2,
                "produced": {"A": 0, "B": 0, "C": 0},
                "purchase_cost": 0,
            }
        ]

        rc = compute_compliance(
            _summary(
                payments=[{"week": 1, "amount": 10}, {"week": 3, "amount": 20}],
                production=production,
            ),
            terms,
            config,
        )
        rcc = compute_compliance(
            _summary(
                payments=[{"week": 1, "amount": 10}, {"week": 3, "amount": 0}],
                production=production,
            ),
            terms,
            config,
        )

        self.assertTrue(rc["customer"]["per_week"][1]["evaluable"])
        self.assertTrue(rc["customer"]["per_week"][1]["compliant"])
        self.assertTrue(rcc["customer"]["per_week"][1]["evaluable"])
        self.assertTrue(rcc["customer"]["per_week"][1]["defected"])
        self.assertEqual(
            compute_behavioral_metrics(rc)["customer"][
                "exploited_compliance_rate"
            ],
            1.0,
        )
        self.assertEqual(
            compute_behavioral_metrics(rcc)["customer"][
                "reciprocal_defection_rate"
            ],
            1.0,
        )

    def test_behavior_distinguishes_unilateral_and_reciprocal_defection(self) -> None:
        compliance = {
            "customer": {
                "per_week": [
                    {"week": 1, "compliant": True, "defected": False},
                    {"week": 3, "compliant": False, "defected": True},
                ]
            },
            "supplier": {
                "per_week": [
                    {"week": 2, "compliant": False, "defected": True},
                ],
                "shortfalls": [],
            },
        }

        behavior = compute_behavioral_metrics(compliance)

        self.assertEqual(behavior["supplier"]["unilateral_defection_rate"], 1.0)
        self.assertIsNone(behavior["supplier"]["reciprocal_defection_rate"])
        self.assertEqual(behavior["customer"]["unilateral_defection_rate"], 0.0)
        self.assertEqual(behavior["customer"]["reciprocal_defection_rate"], 1.0)
    def test_analyze_run_reports_supplier_profit_from_environment_balance(self) -> None:
        terms = {
            "dish_prices": {"A": 12, "B": 7, "C": 5},
            "payment_schedule": [{"week": 1, "amount": 50}],
            "production_schedule": [{"week": 2, "A": 1, "B": 2, "C": 1}],
        }
        summary = _summary(
            payments=[{"week": 1, "amount": 50}],
            production=[
                {
                    "week": 2,
                    "produced": {"A": 1, "B": 2, "C": 1},
                    "purchase_cost": 30,
                }
            ],
        )

        row = analyze_run(summary, terms, {"contingencies": []})

        self.assertEqual(row["supplier_utility"], 20.0)
        self.assertEqual(row["customer_utility"], 127.0)
        self.assertEqual(row["supplier_compliance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
