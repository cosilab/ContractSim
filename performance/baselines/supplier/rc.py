"""Rational Complier (RC) supplier baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._core import SupplierBaseline


class SupplierRC(SupplierBaseline):
    """Supplier that follows the solved policy regardless of customer default."""

    def __init__(
        self,
        contract_id: str,
        contract_dir: str | Path,
        config: dict[str, Any],
        terms: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__("rc", contract_id, Path(contract_dir), config, terms, **kwargs)


__all__ = ("SupplierRC",)

