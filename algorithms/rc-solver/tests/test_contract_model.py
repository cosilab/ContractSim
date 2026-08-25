import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from contract import Contract, PaymentWeek, ProductionWeek
from tests.fixtures import (
    ROLLOVER_CONTRACT_PATH,
    load_rollover_contract,
)


class ContractTests(unittest.TestCase):
    def test_loads_the_rollover_terms_on_the_contract_calendar(self):
        contract = Contract(load_rollover_contract())

        self.assertEqual(contract.products, ("A", "B", "C"))
        self.assertEqual(contract.production_weeks, (2, 4, 6, 8, 10))
        self.assertEqual(contract.payment_weeks, (1, 3, 5, 7, 9, 11))
        self.assertEqual(contract.calendar_weeks, tuple(range(1, 12)))
        self.assertEqual(contract.number_of_weeks, 11)
        self.assertTrue(
            all(isinstance(entry, ProductionWeek) for entry in contract.production_schedule)
        )
        self.assertTrue(
            all(isinstance(entry, PaymentWeek) for entry in contract.payment_schedule)
        )
        self.assertEqual(
            dict(contract.production_schedule[0].product_quantities),
            {"A": 2, "B": 2, "C": 1},
        )
        self.assertEqual(
            dict(contract.product_prices),
            {"A": 11.0, "B": 7.0, "C": 2.0},
        )
        self.assertEqual(
            [payment.amount for payment in contract.payment_schedule],
            [30.0, 30.0, 40.0, 40.0, 40.0, 20.0],
        )

    def test_loads_directly_from_json(self):
        contract = Contract.from_json(ROLLOVER_CONTRACT_PATH)

        self.assertEqual(contract.production_schedule[-1].week, 10)
        self.assertEqual(contract.payment_schedule[-1], PaymentWeek(week=11, amount=20))

    def test_sorts_valid_schedule_rows_by_calendar_week(self):
        source = load_rollover_contract()
        source["production_schedule"].reverse()
        source["payment_schedule"].reverse()

        contract = Contract(source)

        self.assertEqual(contract.production_weeks, (2, 4, 6, 8, 10))
        self.assertEqual(contract.payment_weeks, (1, 3, 5, 7, 9, 11))

    def test_infers_a_shorter_calendar_from_the_contract(self):
        source = load_rollover_contract()
        source["production_schedule"] = source["production_schedule"][:2]
        source["payment_schedule"] = source["payment_schedule"][:3]

        contract = Contract(source)

        self.assertEqual(contract.production_weeks, (2, 4))
        self.assertEqual(contract.payment_weeks, (1, 3, 5))
        self.assertEqual(contract.calendar_weeks, (1, 2, 3, 4, 5))
        self.assertEqual(contract.number_of_weeks, 5)

    def test_is_immutable_and_defensively_copies_nested_terms(self):
        source = load_rollover_contract()
        contract = Contract(source)

        source["production_schedule"][0]["product_quantities"]["A"] = 999
        source["product_prices"]["A"] = 999
        source["payment_schedule"][0]["amount"] = 999

        self.assertEqual(contract.production_schedule[0].product_quantities["A"], 2)
        self.assertEqual(contract.product_prices["A"], 11)
        self.assertEqual(contract.payment_schedule[0].amount, 30)
        with self.assertRaises(TypeError):
            contract.production_schedule[0].product_quantities["A"] = 999
        with self.assertRaises(FrozenInstanceError):
            contract.products = ()

    def test_rejects_incorrect_week_parity(self):
        source = load_rollover_contract()
        source["production_schedule"][0]["week"] = 3
        with self.assertRaisesRegex(ValueError, "production_schedule weeks must be even"):
            Contract(source)

        source = load_rollover_contract()
        source["payment_schedule"][0]["week"] = 2
        with self.assertRaisesRegex(ValueError, "payment_schedule weeks must be odd"):
            Contract(source)

    def test_rejects_duplicate_weeks_and_non_arithmetic_calendars(self):
        source = load_rollover_contract()
        source["payment_schedule"][1]["week"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate weeks"):
            Contract(source)

        source = load_rollover_contract()
        source["payment_schedule"][-1]["week"] = 13
        with self.assertRaisesRegex(ValueError, "arithmetic progression"):
            Contract(source)

    def test_requires_a_payment_before_every_production_and_a_final_payment(self):
        source = load_rollover_contract()
        source["payment_schedule"].pop()
        with self.assertRaisesRegex(ValueError, "one final payment"):
            Contract(source)

        source = load_rollover_contract()
        for payment in source["payment_schedule"]:
            payment["week"] += 2
        with self.assertRaisesRegex(
            ValueError,
            "begin with a payment.*end with a final payment",
        ):
            Contract(source)

    def test_rejects_inconsistent_products_and_invalid_values(self):
        source = load_rollover_contract()
        del source["product_prices"]["C"]
        with self.assertRaisesRegex(ValueError, "products must match"):
            Contract(source)

        source = load_rollover_contract()
        del source["production_schedule"][1]["product_quantities"]["C"]
        with self.assertRaisesRegex(ValueError, "same set of products"):
            Contract(source)

        invalid_fields = (
            ("product_quantities", "A", -1),
            ("product_quantities", "A", 1.5),
        )
        for field, product, value in invalid_fields:
            with self.subTest(field=field, value=value):
                source = load_rollover_contract()
                source["production_schedule"][0][field][product] = value
                with self.assertRaises(ValueError):
                    Contract(source)

        source = load_rollover_contract()
        source["product_prices"]["A"] = -1
        with self.assertRaisesRegex(ValueError, "product_prices"):
            Contract(source)

        source = load_rollover_contract()
        source["payment_schedule"][0]["amount"] = -1
        with self.assertRaisesRegex(ValueError, "amount"):
            Contract(source)

    def test_only_accepts_json_shaped_constructor_input(self):
        valid = Contract(load_rollover_contract())

        with self.assertRaisesRegex(TypeError, "JSON-shaped mapping"):
            Contract(valid)
        with self.assertRaises(TypeError):
            Contract(
                products=valid.products,
                product_prices=valid.product_prices,
                production_schedule=valid.production_schedule,
                payment_schedule=valid.payment_schedule,
            )


if __name__ == "__main__":
    unittest.main()
