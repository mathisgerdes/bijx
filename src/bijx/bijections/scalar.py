"""
One-dimensional bijective transformations for normalizing flows.

This module provides element-wise bijections that can be composed to build
complex normalizing flows. Each bijection implements forward/reverse transforms
with automatic log-Jacobian computation for density estimation.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import ParamSpec, ShapeInfo, default_wrap
from .base import Bijection


def sum_log_jac(x, log_density, log_jac):
    """Sum log-Jacobian over event dimensions for density updates.

    Computes the updated log-density by summing the log-Jacobian contributions
    over the event dimensions while preserving batch dimensions.
    """
    event_dim = jnp.ndim(x) - jnp.ndim(log_density)
    si = ShapeInfo(event_dim=event_dim, channel_dim=0)
    _, si = si.process_event(jnp.shape(x))
    return log_density + jnp.sum(log_jac, axis=si.event_axes)


class OneDimensional(Bijection):
    """Base class for element-wise one-dimensional bijections.

    Subclasses must implement:
    - log_jac(x, y): Log absolute determinant of Jacobian
    - fwd(x): Forward transformation x → y
    - rev(y): Reverse transformation y → x

    The forward/reverse methods handle log-density updates automatically.
    """

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
    """Gaussian CDF normalization with learnable location and scale.

    Type: [-∞, ∞] → [0, 1]
    Transform: Φ((x - μ)/σ) where Φ is the standard Gaussian CDF.
    """

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
    """Tangent-based unbounded transform.

    Type: [0, 1] → [-∞, ∞]
    Transform: tan(π(x - 0.5)) with appropriate scaling.
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(jnp.pi * (1 + y**2)))

    def fwd(self, x, **kwargs):
        return jnp.tan(jnp.pi * (x + 0.5))

    def rev(self, y, **kwargs):
        return jnp.arctan(y) / jnp.pi + 0.5


class SigmoidLayer(OneDimensional):
    """Sigmoid normalization transform.

    Type: [-∞, ∞] → [0, 1]
    Transform: 1/(1 + exp(-x)) with stable numerics.
    """

    def log_jac(self, x, y):
        return jnp.log(y) + jnp.log(1 - y)

    def fwd(self, x, **kwargs):
        return nnx.sigmoid(x)

    def rev(self, y, **kwargs):
        return jnp.log(y / (1 - y))


class TanhLayer(OneDimensional):
    """Hyperbolic tangent bounded transform.

    Type: [-∞, ∞] → [-1, 1]
    Transform: tanh(x) with exact inverse arctanh(y).
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(1 - y**2))

    def fwd(self, x, **kwargs):
        return jnp.tanh(x)

    def rev(self, y, **kwargs):
        return jnp.arctanh(y)


class ExpLayer(OneDimensional):
    """Exponential transform to positive reals.

    Type: [-∞, ∞] → [0, ∞]
    Transform: exp(x) with log-Jacobian equal to x.
    """

    def log_jac(self, x, y, **kwargs):
        return x

    def fwd(self, x, **kwargs):
        return jnp.exp(x)

    def rev(self, y, **kwargs):
        return jnp.log(y)


class SoftPlusLayer(OneDimensional):
    """Numerically stable exponential transform.

    Type: [-∞, ∞] → [0, ∞]
    Transform: log(1 + exp(x)) with stable computation for large |x|.
    """

    def log_jac(self, x, y, **kwargs):
        return -nnx.softplus(-x)

    def fwd(self, x, **kwargs):
        return nnx.softplus(x)

    def rev(self, y, **kwargs):
        return jnp.log(-jnp.expm1(-y)) + y


class PowerLayer(OneDimensional):
    """Power transformation for positive values.

    Type: [0, ∞] → [0, ∞]
    Transform: x^p. Requires strictly positive inputs.
    """

    def __init__(self, exponent: float, *, rngs=None):
        self.exponent = exponent

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(self.exponent)) + (self.exponent - 1) * jnp.log(x)

    def fwd(self, x, **kwargs):
        return x**self.exponent

    def rev(self, y, **kwargs):
        return y ** (1 / self.exponent)


class AffineLayer(OneDimensional):
    """Learnable affine transformation.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: scale * x + shift with learnable parameters.
    """

    def __init__(
        self,
        init_log_scale: ParamSpec = jnp.zeros(()),
        init_shift: ParamSpec = jnp.zeros(()),
        *,
        rngs: nnx.Rngs = None,
    ):
        self.log_scale = default_wrap(init_log_scale, rngs=rngs)
        self.shift = default_wrap(init_shift, rngs=rngs)

    def log_jac(self, x, y, **kwargs):
        return jnp.broadcast_to(self.log_scale.value, x.shape)

    def fwd(self, x, **kwargs):
        scale = jnp.exp(self.log_scale.value)
        return scale * x + self.shift.value

    def rev(self, y, **kwargs):
        scale = jnp.exp(self.log_scale.value)
        return (y - self.shift.value) / scale


class BetaStretch(OneDimensional):
    """Beta-inspired stretching on unit interval.

    Type: [0, 1] → [0, 1]
    Transform: x^a / (x^a + (1-x)^a). Requires a ≠ 0 and valid range.
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
    "ExpLayer",
    "SoftPlusLayer",
    "PowerLayer",
    "AffineLayer",
    "BetaStretch",
]
