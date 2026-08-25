from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from performance.contracts import (
    contract_environment_id,
    validate_contract_environment,
)
from performance.orchestration.run import _contract_dirs


def _write_contract(
    root: Path,
    name: str,
    environment_id: int,
    contingency_variant: str | None = None,
) -> Path:
    contract = root / name
    contract.mkdir()
    (contract / "contract.txt").write_text("contract", encoding="utf-8")
    (contract / "terms.json").write_text("{}", encoding="utf-8")
    config = {"environment_id": environment_id}
    if contingency_variant is not None:
        config["contingency_variant"] = contingency_variant
    (contract / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return contract


class ContractEnvironmentTests(unittest.TestCase):
    def test_reads_and_validates_contract_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _write_contract(Path(temporary), "contract-1", 3)
            self.assertEqual(contract_environment_id(contract), 3)
            validate_contract_environment(contract, 3)
            with self.assertRaisesRegex(ValueError, "targets environment 3"):
                validate_contract_environment(contract, 2)

    def test_batch_selects_only_the_requested_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matching = _write_contract(root, "matching", 2)
            _write_contract(root, "different", 4)
            args = Namespace(mode="batch", contracts_root=root)
            self.assertEqual(_contract_dirs(args, 2), [matching.resolve()])

    def test_batch_excludes_only_the_two_hardcoded_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retained = [
                _write_contract(root, "grim", 1, "grim_trigger"),
                _write_contract(
                    root,
                    "payded-rollover",
                    1,
                    "payment_deduction_rollover",
                ),
                _write_contract(
                    root,
                    "sub-payded-rollover",
                    1,
                    "substitution_payment_deduction_rollover",
                ),
            ]
            _write_contract(root, "payded", 1, "payment_deduction")
            _write_contract(
                root,
                "sub-payded",
                1,
                "substitution_payment_deduction",
            )
            args = Namespace(mode="batch", contracts_root=root)

            self.assertEqual(
                _contract_dirs(args, 1),
                sorted(path.resolve() for path in retained),
            )

    def test_single_rejects_an_environment_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _write_contract(Path(temporary), "contract-1", 3)
            args = Namespace(mode="single", contract_dir=contract)
            with self.assertRaisesRegex(ValueError, "targets environment 3"):
                _contract_dirs(args, 2)

    def test_missing_environment_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = _write_contract(Path(temporary), "contract-1", 1)
            (contract / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declare an integer"):
                contract_environment_id(contract)


if __name__ == "__main__":
    unittest.main()
