"""Rational Conditional Complier (RCC) customer baseline."""

from __future__ import annotations

from typing import Any

from ._core import CustomerBaseline


class CustomerRCC(CustomerBaseline):
    """Customer that complies while the supplier cures or avoids default."""

    def __init__(self, config: dict[str, Any], terms: dict[str, Any], **kwargs: Any) -> None:
        super().__init__("rcc", config, terms, **kwargs)


__all__ = ("CustomerRCC",)

