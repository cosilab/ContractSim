from __future__ import annotations

import json
import unittest

from negotiation.logic.responses import (
    parse_negotiation_response,
    strip_name_prefix,
)


class NegotiationResponseTests(unittest.TestCase):
    def test_parses_structured_response_and_contract(self) -> None:
        response, contract, accept, terminate, meta = parse_negotiation_response(
            '{"response":"Offer","contract":"Pizza: $10\\nWeek 1: 2"}'
        )
        self.assertEqual(response, "Offer")
        self.assertEqual(contract, "Pizza: $10\nWeek 1: 2")
        self.assertFalse(accept)
        self.assertFalse(terminate)
        self.assertTrue(meta["schema_valid"])

    def test_rejects_non_string_action_fields(self) -> None:
        for raw in (
            '{"response":"offer","contract":42}',
            '{"response":"offer","contract":{"price":20}}',
            '{"response":42,"contract":"terms"}',
        ):
            with self.subTest(raw=raw):
                response, contract, accept, terminate, meta = (
                    parse_negotiation_response(raw)
                )
                self.assertIsNone(response)
                self.assertIsNone(contract)
                self.assertFalse(accept)
                self.assertFalse(terminate)
                self.assertFalse(meta["schema_valid"])

    def test_normalizes_contract_newlines_and_outer_whitespace(self) -> None:
        raw = json.dumps(
            {
                "response": "Offer",
                "contract": "  Pizza: $10\\nWeek 1: $10  ",
            }
        )

        _response, contract, _accept, _terminate, _meta = (
            parse_negotiation_response(raw)
        )

        self.assertEqual(contract, "Pizza: $10\nWeek 1: $10")

    def test_whitespace_only_contract_is_absent(self) -> None:
        parsed = parse_negotiation_response(
            '{"response":"Offer","contract":"   "}'
        )

        self.assertIsNone(parsed[1])
        self.assertEqual(parsed[4]["method"], "whole_output")

    def test_accepts_markdown_fenced_json(self) -> None:
        parsed = parse_negotiation_response(
            '```json\n{"response":"Offer","contract":"Pizza: $10"}\n```'
        )

        self.assertEqual(parsed[0], "Offer")
        self.assertEqual(parsed[1], "Pizza: $10")
        self.assertEqual(parsed[4]["method"], "whole_output")

    def test_malformed_json_cannot_trigger_actions(self) -> None:
        for raw in (
            '{"response":"Accept","accept":true,}',
            'preface {"response":"Leaving","terminate":true}',
        ):
            with self.subTest(raw=raw):
                response, contract, accept, terminate, meta = (
                    parse_negotiation_response(raw)
                )
                self.assertIsNone(response)
                self.assertIsNone(contract)
                self.assertFalse(accept)
                self.assertFalse(terminate)
                self.assertEqual(meta["method"], "invalid_json")

    def test_valid_json_requires_response_object(self) -> None:
        for raw in ('["hello"]', '{"accept":true}'):
            with self.subTest(raw=raw):
                response, contract, accept, terminate, meta = (
                    parse_negotiation_response(raw)
                )
                self.assertIsNone(response)
                self.assertIsNone(contract)
                self.assertFalse(accept)
                self.assertFalse(terminate)
                self.assertFalse(meta["schema_valid"])

    def test_strips_only_the_callers_name_prefix(self) -> None:
        self.assertEqual(strip_name_prefix("customer", "Customer: hello"), "hello")
        self.assertEqual(strip_name_prefix("customer", "caterer: hello"), "caterer: hello")


if __name__ == "__main__":
    unittest.main()
