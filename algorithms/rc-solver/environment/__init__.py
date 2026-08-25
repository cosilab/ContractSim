"""Environment domain models and declarative stochastic distributions."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .customer import CustomerSpec
    from .environment import Environment
    from .probability_dist import (
        DeliveryDistributionSpec,
        PriceDistributionSpec,
        PriceOutcome,
        ProbabilityDist,
        SpoilageDistributionSpec,
        SpoilageOutcome,
    )
    from .supplier import SupplierSpec

__all__ = [
    "CustomerSpec",
    "DeliveryDistributionSpec",
    "Environment",
    "PriceDistributionSpec",
    "PriceOutcome",
    "ProbabilityDist",
    "SpoilageDistributionSpec",
    "SpoilageOutcome",
    "SupplierSpec",
    "build_probability_dist",
]


def __getattr__(name: str) -> Any:
    """Load public components lazily so submodules remain executable with ``-m``."""
    if name == "CustomerSpec":
        from .customer import CustomerSpec

        return CustomerSpec
    if name == "SupplierSpec":
        from .supplier import SupplierSpec

        return SupplierSpec
    if name == "Environment":
        from .environment import Environment

        return Environment
    probability_names = {
        "DeliveryDistributionSpec",
        "PriceDistributionSpec",
        "PriceOutcome",
        "ProbabilityDist",
        "SpoilageDistributionSpec",
        "SpoilageOutcome",
        "build_probability_dist",
    }
    if name in probability_names:
        from .probability_dist import (
            DeliveryDistributionSpec,
            PriceDistributionSpec,
            PriceOutcome,
            ProbabilityDist,
            SpoilageDistributionSpec,
            SpoilageOutcome,
            build_probability_dist,
        )

        return {
            "DeliveryDistributionSpec": DeliveryDistributionSpec,
            "PriceDistributionSpec": PriceDistributionSpec,
            "PriceOutcome": PriceOutcome,
            "ProbabilityDist": ProbabilityDist,
            "SpoilageDistributionSpec": SpoilageDistributionSpec,
            "SpoilageOutcome": SpoilageOutcome,
            "build_probability_dist": build_probability_dist,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
