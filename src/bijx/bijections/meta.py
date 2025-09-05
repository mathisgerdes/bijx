"""
Meta-transformations for shape manipulation without density changes.

This module provides bijections that manipulate tensor shapes and dimensions
without affecting the underlying data distribution.
The :class:`MetaLayer` class can be used to conveniently define new
meta-transformations, that do not change the density.

Key components:
- MetaLayer: Base class for density-preserving transformations
- ExpandDims/SqueezeDims: Dimension manipulation utilities
- Reshape: Flexible tensor reshaping with shape validation
- Partial: Bijection wrapper with fixed keyword arguments
"""

import typing as tp
from functools import partial

import jax.numpy as jnp

from .base import Bijection


class MetaLayer(Bijection):
    """Convenient constructor for bijections that preserve probability density.

    Example:
        >>> transpose = MetaLayer(
        ...     forward=lambda x: x[..., ::-1],
        ...     reverse=lambda x: x[..., ::-1],
        ... )
        >>> y, log_det = transpose.forward(x, log_density)
        >>> # log_det unchanged, y has transposed last two dims


    Args:
        forward: Map ``x -> y`` that does not change the density.
        reverse: Map ``y -> x`` that does not change the density.
        rngs: Included for compatibility; not used.
    """

    def __init__(self, forward: tp.Callable, reverse: tp.Callable, *, rngs=None):
        self._forward = forward
        self._reverse = reverse

    def forward(self, x, log_density):
        return self._forward(x), log_density

    def reverse(self, x, log_density):
        return self._reverse(x), log_density


class ExpandDims(MetaLayer):
    """Expand tensor dimensions along specified axis.

    Adds a singleton dimension at the specified axis position.

    Example:
        >>> expand = ExpandDims(axis=-1)
        >>> x = jnp.array([[1, 2], [3, 4]])  # Shape (2, 2)
        >>> y, log_det = expand.forward(x, log_density)
        >>> # y has shape (2, 2, 1), log_det unchanged

    Args:
        axis: Axis along which to expand dimensions.
        rngs: Included for compatibility; not used.
    """

    def __init__(self, axis: int = -1, *, rngs=None):
        super().__init__(
            partial(jnp.expand_dims, axis=axis),
            partial(jnp.squeeze, axis=axis),
        )


class SqueezeDims(MetaLayer):
    """Remove singleton dimensions along specified axis.

    Removes dimensions of size 1 at the specified axis position.
    This is the inverse operation of :class:`ExpandDims`.

    Type: Shape(..., d_k, ..., 1, ...) → Shape(..., d_k, ...)
    Transform: Remove dimension of size 1 at specified axis

    Example:
        >>> squeeze = SqueezeDims(axis=-1)
        >>> x = jnp.array([[[1], [2]], [[3], [4]]])  # Shape (2, 2, 1)
        >>> y, log_det = squeeze.forward(x, log_density)
        >>> # y has shape (2, 2), log_det unchanged

    Args:
        axis: Axis along which to squeeze dimensions.
        rngs: Random number generators (unused).

    """

    def __init__(self, axis: int = -1, *, rngs=None):
        super().__init__(
            partial(jnp.squeeze, axis=axis),
            partial(jnp.expand_dims, axis=axis),
        )


class Reshape(MetaLayer):
    """Reshape tensor event dimensions while preserving batch dimensions.

    Reshapes the event portion of tensors from one shape to another,
    preserving all batch dimensions. The total number of elements in
    the event shape must remain constant.

    Type: Batch + from_shape → Batch + to_shape
    Transform: Reshape event dimensions only

    Key features:
        - Batch dimensions are automatically preserved
        - Event shape compatibility is validated
        - Bidirectional reshaping with shape memory

    Example:
        >>> reshape = Reshape(from_shape=(4, 4), to_shape=(16,))
        >>> x = jnp.ones((3, 4, 4))  # Batch size 3, event shape (4, 4)
        >>> y, log_det = reshape.forward(x, log_density)
        >>> # y has shape (3, 16), log_det unchanged

    Args:
        from_shape: Original event shape to reshape from.
        to_shape: Target event shape to reshape to.
        rngs: Included for compatibility; not used.

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
    """Bijection wrapper that fixes keyword arguments.

    Creates a new bijection by partially applying keyword arguments to
    an existing bijection. This is useful for creating specialized versions
    of configurable bijections or for setting default parameters.

    The wrapped bijection receives both the fixed kwargs and any additional
    kwargs passed at call time, with call-time kwargs taking precedence.
    """

    def __init__(self, bijection: Bijection, **kwargs):
        """Initialize partial bijection with fixed keyword arguments.

        Args:
            bijection: The bijection to wrap.
            **kwargs: Keyword arguments to fix for all calls.
        """
        self.bijection = bijection
        self.kwargs = kwargs

    def _kwargs(self, kwargs):
        """Update default kwargs with additional kwargs."""
        full_kwargs = self.kwargs.copy()
        full_kwargs.update(kwargs)
        return full_kwargs

    def forward(self, x, log_density, **kwargs):
        """Apply forward transformation with fixed and additional kwargs.

        Args:
            x: Input to transform.
            log_density: Input log density.
            **kwargs: Additional keyword arguments (override fixed ones).

        Returns:
            Result from wrapped bijection's forward method.
        """
        return self.bijection.forward(x, log_density, **self._kwargs(kwargs))

    def reverse(self, x, log_density, **kwargs):
        """Apply reverse transformation with fixed and additional kwargs.

        Args:
            x: Input to inverse transform.
            log_density: Input log density.
            **kwargs: Additional keyword arguments (override fixed ones).

        Returns:
            Result from wrapped bijection's reverse method.
        """
        return self.bijection.reverse(x, log_density, **self._kwargs(kwargs))
