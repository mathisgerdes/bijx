"""
Linear and affine transformations.
"""

import typing as tp

import flax.typing as ftp
import jax.numpy as jnp
from flax import nnx

from ..utils import ParamSpec, default_wrap
from .base import Bijection


class Scaling(Bijection):
    def __init__(
        self,
        shape_or_val: ParamSpec = (),
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
        shape_or_val: ParamSpec = (),
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


class LinearAffine(Bijection):
    def __init__(
        self,
        scale: ParamSpec = (),
        shift: ParamSpec = (),
        transform_scale: tp.Callable = lambda x: x,
        transform_shift: tp.Callable = jnp.exp,
        *,
        init_scale: ftp.Initializer = nnx.initializers.zeros,
        init_shift: ftp.Initializer = nnx.initializers.zeros,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform_scale = transform_scale
        self.transform_shift = transform_shift
        self.scale_val = default_wrap(scale, init_fn=init_scale, rngs=rngs)
        self.shift_val = default_wrap(shift, init_fn=init_shift, rngs=rngs)

    @property
    def scale(self):
        scale = self.scale_val.value
        return self.transform_scale(scale)

    @property
    def shift(self):
        shift = self.shift_val.value
        return self.transform_shift(shift)

    def forward(self, x, log_density, **kwargs):
        scale, shift = self.scale, self.shift
        return x * scale + shift, log_density

    def reverse(self, x, log_density, **kwargs):
        scale, shift = self.scale, self.shift
        return (x - shift) / scale, log_density
