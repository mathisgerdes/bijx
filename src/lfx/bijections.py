import typing as tp
from functools import partial

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils import Const, default_wrap


class Bijection(nnx.Module):
    def forward(self, x, log_density, **kwargs):
        return x, log_density

    def reverse(self, x, log_density, **kwargs):
        return x, log_density

    def __call__(self, x, log_density, **kwargs):
        return self.forward(x, log_density, **kwargs)

    def invert(self):
        return Inverse(self)


class Inverse(Bijection):
    def __init__(self, bijection: Bijection):
        self.bijection = bijection

    def forward(self, x, log_density, **kwargs):
        return self.bijection.reverse(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.bijection.forward(x, log_density, **kwargs)


class Chain(Bijection):
    def __init__(self, *bijections: Bijection):
        self.bijections = list(bijections)

    def forward(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(self.bijections, arg_list, strict=True):
            x, log_density = bijection.forward(x, log_density, *args, **kwargs)
        return x, log_density

    def reverse(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(
            reversed(self.bijections), reversed(arg_list), strict=True
        ):
            x, log_density = bijection.reverse(x, log_density, *args, **kwargs)
        return x, log_density


class ScanChain(Bijection):

    def __init__(self, stack):
        self.stack = stack

    def _forward(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.forward(*carry, **kwargs), None

    def _reverse(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.reverse(*carry, **kwargs), None

    def forward(self, x, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (y, lp), _ = jax.lax.scan(
            partial(self._forward, graph=graph, **kwargs),
            (x, log_density),
            variables,
        )
        return y, lp

    def reverse(self, y, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (x, lp), _ = jax.lax.scan(
            partial(self._reverse, graph=graph, **kwargs),
            (y, log_density),
            variables,
            reverse=True,
        )
        return x, lp


class Frozen(Bijection):
    def __init__(self, bijection: Bijection):
        self.frozen = bijection

    def forward(self, x, log_density, **kwargs):
        return self.frozen.forward(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.frozen.reverse(x, log_density, **kwargs)


class Scaling(Bijection):
    def __init__(
        self,
        shape_or_val: jax.Array | nnx.Variable | tuple[int, ...],
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.ones,
        dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        if isinstance(shape_or_val, tuple):
            self.scale_val = nnx.Param(init(rngs.params(), shape_or_val, dtype))
        else:
            self.scale_val = default_wrap(shape_or_val, Const)

    @property
    def scale(self):
        try:
            scale = self.scale_val.value
        except AttributeError:
            scale = self.scale_val
        return self.transform(scale)

    def forward(self, x, log_density, **kwargs):
        return x * self.scale, log_density - jnp.sum(jnp.log(jnp.abs(self.scale)))

    def reverse(self, x, log_density, **kwargs):
        return x / self.scale, log_density + jnp.sum(jnp.log(jnp.abs(self.scale)))


class Shift(Bijection):
    def __init__(
        self,
        shape_or_val: jax.Array | tuple[int, ...],
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.zeros,
        dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        if isinstance(shape_or_val, jax.Array | np.ndarray | int | float):
            self.shift_val = Const(shape_or_val)
        else:
            if rngs is None:
                raise ValueError(
                    "rngs must be provided if shape_or_val is not a constant"
                )
            self.shift_val = nnx.Param(init(rngs.params(), shape_or_val, dtype))

    @property
    def shift(self):
        try:
            shift = self.shift_val.value
        except AttributeError:
            shift = self.shift_val
        return self.transform(shift)

    def forward(self, x, log_density, **kwargs):
        return x + self.shift, log_density

    def reverse(self, x, log_density, **kwargs):
        return x - self.shift, log_density


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
