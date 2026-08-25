from __future__ import annotations

import unittest
from itertools import product

from performance.baselines.customer import CustomerRC, CustomerRCC
from performance.engine.environment import DISHES, load_performance_environment


PRICES = {"A": 10, "B": 6, "C": 3}


def terms() -> dict:
    return {
        "dish_prices": dict(PRICES),
        "payment_schedule": [
            {"week": 3, "amount": 25},
            {"week": 11, "amount": 25},
        ],
        "production_schedule": [
            {"week": 2, "A": 0, "B": 0, "C": 0},
            {"week": 10, "A": 0, "B": 0, "C": 0},
        ],
    }


class CustomerContingencyTests(unittest.TestCase):
    def test_rc_and_rcc_apply_payment_deductions(self) -> None:
        contract_terms = terms()
        contract_terms["production_schedule"][0].update(
            {"A": 1, "B": 1, "C": 1}
        )
        config = {
            "contingencies": ["payment_deduction"],
            "contingency_params": {
                "payment_deduction": {"deduction_prices": PRICES}
            },
        }
        rc = CustomerRC(config, contract_terms, environment_id=1)
        rcc = CustomerRCC(config, contract_terms, environment_id=1)

        for customer in (rc, rcc):
            customer.observe_production(
                week=2,
                produced={"A": 0, "B": 0, "C": 0},
            )

        rc_payment, rc_decision = rc.decide_payment(
            week=3,
            remaining_budget=25,
        )
        rcc_payment, rcc_decision = rcc.decide_payment(
            week=3,
            remaining_budget=25,
        )

        self.assertEqual(rc_payment, 6)
        self.assertEqual(rc_decision["reason"], "rc_payment_deduction")
        self.assertEqual(rcc_payment, 6)
        self.assertEqual(rcc_decision["reason"], "rcc_payment_deduction")
        self.assertEqual(rc.pending_deduction, 0)
        self.assertEqual(rcc.pending_deduction, 0)

    def test_substitution_only_matches_the_complete_coverage_rule(self) -> None:
        environment = load_performance_environment(1)
        minimums = environment.customer_minimum_quantities
        points = environment.soup_equivalents
        customer = CustomerRCC(
            {"contingencies": ["substitution"]},
            terms(),
            environment_id=1,
        )

        for target_values in product(range(5), repeat=3):
            target = dict(zip(DISHES, target_values))
            exact = {
                dish: min(target[dish], minimums[dish])
                for dish in DISHES
            }
            demand_points = sum(
                (target[dish] - exact[dish]) * points[dish]
                for dish in DISHES
            )
            customer.production_schedule[2] = target

            for produced_values in product(range(7), repeat=3):
                produced = dict(zip(DISHES, produced_values))
                exact_met = all(
                    produced[dish] >= exact[dish]
                    for dish in DISHES
                )
                supply_points = sum(
                    max(produced[dish] - exact[dish], 0) * points[dish]
                    for dish in DISHES
                )
                expected_satisfied = exact_met and supply_points >= demand_points

                customer.suspended = False
                customer.carry_deficit = {dish: 0 for dish in DISHES}
                customer.history.clear()
                record = customer.observe_production(week=2, produced=produced)

                self.assertEqual(record["minima_met"], exact_met)
                self.assertEqual(record["breach"], not expected_satisfied)
                self.assertEqual(customer.suspended, not expected_satisfied)

    def test_payment_deduction_only_matches_the_grace_and_cure_rule(self) -> None:
        customer = CustomerRCC(
            {
                "contingencies": ["payment_deduction"],
                "contingency_params": {
                    "payment_deduction": {"deduction_prices": PRICES}
                },
            },
            terms(),
            environment_id=1,
        )

        for target_values in product(range(5), repeat=3):
            target = dict(zip(DISHES, target_values))
            customer.production_schedule[2] = target

            for produced_values in product(range(6), repeat=3):
                produced = dict(zip(DISHES, produced_values))
                shortfall = {
                    dish: max(target[dish] - produced[dish], 0)
                    for dish in DISHES
                }
                breach = any(shortfall.values())
                deduction = sum(
                    shortfall[dish] * PRICES[dish]
                    for dish in DISHES
                )

                for prior_shortfall in (False, True):
                    for scheduled_payment in (0, 5, 25, 100):
                        customer.payment_schedule[3] = scheduled_payment
                        customer.suspended = False
                        customer.had_prior_shortfall = prior_shortfall
                        customer.pending_deduction = 0
                        customer.carry_deficit = {
                            dish: 0 for dish in DISHES
                        }
                        customer.total_paid = 0
                        customer.history.clear()

                        record = customer.observe_production(
                            week=2,
                            produced=produced,
                        )
                        payment, _decision = customer.decide_payment(
                            week=3,
                            remaining_budget=1_000,
                        )

                        expected_suspended = prior_shortfall and breach
                        expected_payment = (
                            0
                            if expected_suspended
                            else max(scheduled_payment - deduction, 0)
                        )
                        self.assertEqual(record["shortfall"], shortfall)
                        self.assertEqual(
                            record["pending_deduction"],
                            deduction if breach else 0,
                        )
                        self.assertEqual(customer.suspended, expected_suspended)
                        self.assertEqual(customer.had_prior_shortfall, breach)
                        self.assertEqual(payment, expected_payment)

    def test_incomplete_substitution_uses_physical_deduction(self) -> None:
        contract_terms = terms()
        contract_terms["production_schedule"][0].update(
            {"A": 4, "B": 1, "C": 1}
        )
        customer = CustomerRCC(
            {
                "contingencies": ["substitution", "payment_deduction"],
                "contingency_params": {
                    "payment_deduction": {"deduction_prices": PRICES}
                },
            },
            contract_terms,
            environment_id=1,
        )

        record = customer.observe_production(
            week=2,
            produced={"A": 1, "B": 3, "C": 1},
        )
        payment, _decision = customer.decide_payment(
            week=3,
            remaining_budget=25,
        )

        self.assertEqual(record["shortfall"], {"A": 3, "B": 0, "C": 0})
        self.assertEqual(record["pending_deduction"], 30)
        self.assertFalse(customer.suspended)
        self.assertEqual(payment, 0)

    def test_rollover_only_matches_ordinary_and_final_round_rules(self) -> None:
        max_deficit = 2
        customer = CustomerRCC(
            {
                "contingencies": ["rollover"],
                "contingency_params": {
                    "rollover": {"max_deficit": max_deficit}
                },
            },
            terms(),
            environment_id=1,
        )

        for target_values in product(range(4), repeat=3):
            target = dict(zip(DISHES, target_values))
            customer.production_schedule[2] = target
            customer.production_schedule[10] = target

            for carry_values in product(range(max_deficit + 1), repeat=3):
                carry = dict(zip(DISHES, carry_values))
                had_carry = any(carry.values())
                effective_target = {
                    dish: target[dish] + carry[dish]
                    for dish in DISHES
                }

                for produced_values in product(range(6), repeat=3):
                    produced = dict(zip(DISHES, produced_values))
                    shortfall = {
                        dish: max(effective_target[dish] - produced[dish], 0)
                        for dish in DISHES
                    }
                    breach = any(shortfall.values())
                    expected_ordinary_suspension = breach and (
                        had_carry
                        or any(
                            shortfall[dish] > max_deficit
                            for dish in DISHES
                        )
                    )

                    for week, expected_suspended in (
                        (2, expected_ordinary_suspension),
                        (10, breach),
                    ):
                        customer.suspended = False
                        customer.carry_deficit = dict(carry)
                        customer.history.clear()
                        record = customer.observe_production(
                            week=week,
                            produced=produced,
                        )

                        self.assertEqual(
                            record["effective_target"],
                            effective_target,
                        )
                        self.assertEqual(record["shortfall"], shortfall)
                        self.assertEqual(customer.carry_deficit, shortfall)
                        self.assertEqual(
                            customer.suspended,
                            expected_suspended,
                        )

        customer.production_schedule[10] = {"A": 1, "B": 1, "C": 1}
        customer.suspended = False
        customer.carry_deficit = {dish: 0 for dish in DISHES}
        customer.observe_production(
            week=10,
            produced={"A": 0, "B": 1, "C": 1},
        )
        payment, _decision = customer.decide_payment(
            week=11,
            remaining_budget=25,
        )
        self.assertTrue(customer.suspended)
        self.assertEqual(payment, 0)


if __name__ == "__main__":
    unittest.main()
