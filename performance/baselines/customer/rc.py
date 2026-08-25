"""Rational Complier (RC) customer baseline."""

from __future__ import annotations

from typing import Any

from ._core import CustomerBaseline


class CustomerRC(CustomerBaseline):
    """Customer that follows the contract regardless of supplier behavior."""

    def __init__(self, config: dict[str, Any], terms: dict[str, Any], **kwargs: Any) -> None:
        super().__init__("rc", config, terms, **kwargs)


__all__ = ("CustomerRC",)
