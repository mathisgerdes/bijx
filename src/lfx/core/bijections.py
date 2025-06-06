import typing as tp
from functools import partial

import flax.typing as ftp
import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import ParamSpec, default_wrap


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
        shape_or_val: ParamSpec,
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.ones,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        self.scale_val = default_wrap(shape_or_val, init_fn=init, rngs=rngs)

    @property
    def scale(self):
        scale = self.scale_val.value
        return self.transform(scale)

    def _broadcast_shapes(self, x, log_density):
        scale = jnp.broadcast_to(self.scale, x.shape)
        batch_dim = jnp.ndim(log_density)
        assert jnp.shape(x)[:batch_dim] == jnp.shape(
            log_density
        ), "batch dimensions of x and log_density must match"
        data_dim = jnp.ndim(x) - batch_dim
        data_axes = tuple(range(batch_dim, batch_dim + data_dim))
        assert (
            x.shape[batch_dim:] == scale.shape[batch_dim:]
        ), "scaling must not change data shape"
        log_jac = jnp.sum(jnp.log(jnp.abs(scale)), axis=data_axes)
        assert (
            log_jac.shape == log_density.shape[:batch_dim]
        ), "log_jac must have same batch dimensions as log_density"
        return scale, log_jac

    def forward(self, x, log_density, **kwargs):
        scale, log_jac = self._broadcast_shapes(x, log_density)
        return x * scale, log_density - log_jac

    def reverse(self, x, log_density, **kwargs):
        scale, log_jac = self._broadcast_shapes(x, log_density)
        return x / scale, log_density + log_jac


class Shift(Bijection):
    def __init__(
        self,
        shape_or_val: ParamSpec,
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.zeros,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        self.shift_val = default_wrap(shape_or_val, init_fn=init, rngs=rngs)

    @property
    def shift(self):
        shift = self.shift_val.value
        return self.transform(shift)

    def forward(self, x, log_density, **kwargs):
        return x + self.shift, log_density

    def reverse(self, x, log_density, **kwargs):
        return x - self.shift, log_density


# Meta transformations (shape-only)


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
