"""Rational Exploiter (RE) supplier baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._core import SupplierBaseline


class SupplierRE(SupplierBaseline):
    """Supplier that exploits execution by never buying or producing."""

    def __init__(
        self,
        contract_id: str,
        contract_dir: str | Path,
        config: dict[str, Any],
        terms: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__("re", contract_id, Path(contract_dir), config, terms, **kwargs)


__all__ = ("SupplierRE",)

