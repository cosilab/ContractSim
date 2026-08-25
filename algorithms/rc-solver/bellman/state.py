"""JAX value tensors carried between production-week Bellman steps."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class WeekValueState(NamedTuple):
    """Bellman values indexed by resource inventories and supplier capital.

    A ``NamedTuple`` is a native JAX pytree, so this state can pass through
    ``jax.jit`` and ``jax.lax.scan`` without custom flattening code.
    """

    satisfaction_probability: jax.Array
    supplier_utility: jax.Array
    customer_value: jax.Array
    total_paid: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.satisfaction_probability.shape

    def stacked(self) -> jax.Array:
        """Return the metric-first representation used by the Bellman operator."""
        return jnp.stack(self, axis=0)

    @classmethod
    def from_stacked(cls, values: jax.Array) -> "WeekValueState":
        if values.ndim < 3 or values.shape[0] != 4:
            raise ValueError("stacked week values must have four metric arrays")
        return cls(*(values[index] for index in range(4)))
