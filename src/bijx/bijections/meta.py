"""
Meta transformations that do not change density or transform data (shape-only).
"""

import typing as tp
from functools import partial

import jax.numpy as jnp

from .base import Bijection


class MetaLayer(Bijection):
    """Convenience class for operations that do not change density."""

    def __init__(self, forward: tp.Callable, reverse: tp.Callable, *, rngs=None):
        self._forward = forward
        self._reverse = reverse

    def forward(self, x, log_density):
        return self._forward(x), log_density

    def reverse(self, x, log_density):
        return self._reverse(x), log_density


class ExpandDims(MetaLayer):
    def __init__(self, axis: int = -1, *, rngs=None):
        super().__init__(
            partial(jnp.expand_dims, axis=axis),
            partial(jnp.squeeze, axis=axis),
        )


class SqueezeDims(MetaLayer):
    def __init__(self, axis: int = -1, *, rngs=None):
        super().__init__(
            partial(jnp.squeeze, axis=axis),
            partial(jnp.expand_dims, axis=axis),
        )


class Reshape(MetaLayer):
    """Reshape arrays.

    Args:
        from_shape: Original event shape to reshape from.
        to_shape: Target event shape to reshape to.
    """

    def __init__(
        self, from_shape: tuple[int, ...], to_shape: tuple[int, ...], *, rngs=None
    ):
        self.from_shape = from_shape
        self.to_shape = to_shape
        super().__init__(self._forward, self._reverse)

    def _forward(self, x):
        shape = jnp.shape(x)
        batch_shape = shape[: -len(self.from_shape)]
        from_shape = shape[-len(self.from_shape) :]
        assert from_shape == self.from_shape
        return jnp.reshape(x, batch_shape + self.to_shape)

    def _reverse(self, x):
        shape = jnp.shape(x)
        batch_shape = shape[: -len(self.to_shape)]
        to_shape = shape[-len(self.to_shape) :]
        assert to_shape == self.to_shape
        return jnp.reshape(x, batch_shape + self.from_shape)


class Partial(Bijection):
    def __init__(self, bijection: Bijection, **kwargs):
        self.bijection = bijection
        self.kwargs = kwargs

    def forward(self, x, log_density, **kwargs):
        return self.bijection(x, log_density, **self.kwargs, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.bijection.reverse(x, log_density, **self.kwargs, **kwargs)
