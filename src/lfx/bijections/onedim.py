"""
Parametric and parameter-free one-dimensional bijections.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import ParamSpec, ShapeInfo, default_wrap
from .base import Bijection


def sum_log_jac(x, log_density, log_jac):
    """Sum divergence over event axes."""
    event_dim = jnp.ndim(x) - jnp.ndim(log_density)
    si = ShapeInfo(event_dim=event_dim, channel_dim=0)
    _, si = si.process_event(jnp.shape(x))
    return log_density + jnp.sum(log_jac, axis=si.event_axes)


class OneDimensional(Bijection):

    def log_jac(self, x, y, **kwargs):
        raise NotImplementedError

    def fwd(self, x, **kwargs):
        raise NotImplementedError()

    def rev(self, y, **kwargs):
        raise NotImplementedError()

    def forward(self, x, log_density, **kwargs):
        y = self.fwd(x, **kwargs)
        return y, sum_log_jac(x, log_density, -self.log_jac(x, y))

    def reverse(self, y, log_density, **kwargs):
        x = self.rev(y, **kwargs)
        return x, sum_log_jac(x, log_density, self.log_jac(x, y))


class GaussianCDF(OneDimensional):
    """Invertible map from [-inf, inf] to [0, 1] using Gaussian cdfs."""

    def __init__(
        self,
        init_log_scale: ParamSpec = jnp.zeros(()),
        init_mean: ParamSpec = jnp.zeros(()),
        *,
        rngs: nnx.Rngs = None,
    ):
        self.mean = default_wrap(init_mean, rngs=rngs)
        self.log_scale = default_wrap(init_log_scale, rngs=rngs)

    def log_jac(self, x, y, **kwargs):
        loc = self.mean.value
        scale = jnp.exp(self.log_scale.value)
        return jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale)

    def fwd(self, x, **kwargs):
        loc = self.mean.value
        scale = jnp.exp(self.log_scale.value)
        return jax.scipy.stats.norm.cdf(x, loc=loc, scale=scale)

    def rev(self, y, **kwargs):
        loc = self.mean.value
        scale = jnp.exp(self.log_scale.value)
        return jax.scipy.stats.norm.ppf(y, loc=loc, scale=scale)


class TanLayer(OneDimensional):
    """Invertible map from [0, 1] to [-inf, inf] using tan/arctan."""

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(jnp.pi * (1 + y**2)))

    def fwd(self, x, **kwargs):
        return jnp.tan(jnp.pi * (x + 0.5))

    def rev(self, y, **kwargs):
        return jnp.arctan(y) / jnp.pi + 0.5


class SigmoidLayer(OneDimensional):
    """Invertible map from [-inf, inf] to [0, 1] using sigmoid."""

    def log_jac(self, x, y):
        return jnp.log(y) + jnp.log(1 - y)

    def fwd(self, x, **kwargs):
        return nnx.sigmoid(x)

    def rev(self, y, **kwargs):
        return jnp.log(y / (1 - y))


class TanhLayer(OneDimensional):
    """Invertible map from [-inf, inf] to [-1, 1] using tanh."""

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(1 - y**2))

    def fwd(self, x, **kwargs):
        return jnp.tanh(x)

    def rev(self, y, **kwargs):
        return jnp.arctanh(y)


class BetaStretch(OneDimensional):
    """Invertible map [0, 1] -> [0, 1] inspired by beta CDFs.

    Note that this module does not check for valid range and a != 0.
    """

    def __init__(self, a: ParamSpec, *, rngs=None):
        self.a = default_wrap(a, rngs=rngs)

    def log_jac(self, x, y, **kwargs):
        a = self.a.value
        return (
            jnp.log(a)
            + jnp.log(x ** (a - 1) * (1 - x) ** a + x**a * (1 - x) ** (a - 1))
            - 2 * jnp.log(x**a + (1 - x) ** a)
        )

    def fwd(self, x, **kwargs):
        a = self.a.value
        xa = x**a
        return xa / (xa + (1 - x) ** a)

    def rev(self, y, **kwargs):
        a = self.a.value
        r = (y / (1 - y)) ** (1 / a)
        return r / (r + 1)

__all__ = [
    "sum_log_jac",
    "OneDimensional",
    "GaussianCDF",
    "TanLayer",
    "SigmoidLayer",
    "TanhLayer",
    "BetaStretch",
]
