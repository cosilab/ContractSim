"""Rational Exploiter (RE) customer baseline."""

from __future__ import annotations

from typing import Any

from ._core import CustomerBaseline


class CustomerRE(CustomerBaseline):
    """Customer that exploits execution by never paying."""

    def __init__(self, config: dict[str, Any], terms: dict[str, Any], **kwargs: Any) -> None:
        super().__init__("re", config, terms, **kwargs)


__all__ = ("CustomerRE",)

