"""Rational Conditional Complier (RCC) supplier baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._core import SupplierBaseline


class SupplierRCC(SupplierBaseline):
    """Supplier that follows policy until the customer defaults."""

    def __init__(
        self,
        contract_id: str,
        contract_dir: str | Path,
        config: dict[str, Any],
        terms: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__("rcc", contract_id, Path(contract_dir), config, terms, **kwargs)


__all__ = ("SupplierRCC",)

