"""
One-dimensional bijective transformations for normalizing flows.

This module provides element-wise bijections that can be composed to build
complex normalizing flows. Each bijection implements forward/reverse transforms
with automatic log-Jacobian computation for density estimation.

All bijections here have an automatic broadcasting behavior:
- Follow standard numpy broadcasting rules, except:
- Automatically infer event shape from log-density vs input shapes
- Sum scalar log-jacobians over event axes

"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct

from ..utils import ParamSpec, ShapeInfo, default_wrap
from .base import Bijection


# not exported (can still be accessed as bijx.bijections.scalar...)
@struct.dataclass
class TransformedParameter:
    param: nnx.Variable
    transform: Callable

    @property
    def value(self):
        transform = self.transform
        if transform is None:
            return self.param.value
        return transform(self.param.value)


_softplus_inv_one = jnp.log(jnp.expm1(1))


# not exported
def sum_log_jac(x, log_density, log_jac):
    """Sum log-Jacobian over event dimensions for density updates.

    Computes the updated log-density by summing the log-Jacobian contributions
    over the event dimensions while preserving batch dimensions.
    """
    event_dim = jnp.ndim(x) - jnp.ndim(log_density)
    si = ShapeInfo(event_dim=event_dim, channel_dim=0)
    _, si = si.process_event(jnp.shape(x))
    return log_density + jnp.sum(log_jac, axis=si.event_axes)


class ScalarBijection(Bijection):
    """Base class for element-wise one-dimensional bijections.

    Subclasses must implement:
    - log_jac(x, y): Log absolute determinant of Jacobian
    - fwd(x): Forward transformation x → y
    - rev(y): Reverse transformation y → x

    The forward/reverse methods handle log-density updates automatically.
    """

    def log_jac(self, x, y, **kwargs):
        raise NotImplementedError()

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


class GaussianCDF(ScalarBijection):
    """Gaussian CDF normalization with learnable location and scale.

    Type: [-∞, ∞] → [0, 1]
    Transform: Φ((x - μ)/σ) where Φ is the standard Gaussian CDF.
    """

    def __init__(
        self,
        scale: ParamSpec = (),
        mean: ParamSpec = (),
        transform_scale: Callable | None = nnx.softplus,
        transform_mean: Callable | None = None,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.mean = TransformedParameter(
            param=default_wrap(mean, init_fn=nnx.initializers.zeros, rngs=rngs),
            transform=transform_mean,
        )
        self.scale = TransformedParameter(
            param=default_wrap(
                scale,
                # initialize such that scale.value = 1
                init_fn=nnx.initializers.constant(_softplus_inv_one),
                rngs=rngs,
            ),
            transform=transform_scale,
        )

    def log_jac(self, x, y, **kwargs):
        return jax.scipy.stats.norm.logpdf(
            x, loc=self.mean.value, scale=self.scale.value
        )

    def fwd(self, x, **kwargs):
        return jax.scipy.stats.norm.cdf(x, loc=self.mean.value, scale=self.scale.value)

    def rev(self, y, **kwargs):
        return jax.scipy.stats.norm.ppf(y, loc=self.mean.value, scale=self.scale.value)


class Tan(ScalarBijection):
    """Tangent-based unbounded transform.

    Type: [0, 1] → [-∞, ∞]
    Transform: tan(π(x - 0.5)) with appropriate scaling.
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(jnp.pi * (1 + y**2)))

    def fwd(self, x, **kwargs):
        return jnp.tan(jnp.pi * (x - 0.5))

    def rev(self, y, **kwargs):
        return jnp.arctan(y) / jnp.pi + 0.5


class Sigmoid(ScalarBijection):
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


class Tanh(ScalarBijection):
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


class Exponential(ScalarBijection):
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


class SoftPlus(ScalarBijection):
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


class Power(ScalarBijection):
    """Power transformation for positive values.

    Type: [0, ∞] → [0, ∞]
    Transform: x^p. Require p > 0.
    """

    def __init__(
        self,
        exponent: ParamSpec = (),
        transform_exponent: Callable | None = jnp.abs,
        *,
        rngs=None,
    ):
        self.exponent = TransformedParameter(
            param=default_wrap(exponent, init_fn=nnx.initializers.ones, rngs=rngs),
            transform=transform_exponent,
        )

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(self.exponent.value)) + (
            self.exponent.value - 1
        ) * jnp.log(x)

    def fwd(self, x, **kwargs):
        return x**self.exponent.value

    def rev(self, y, **kwargs):
        return y ** (1 / self.exponent.value)


class Sinh(ScalarBijection):
    """Hyperbolic sine transformation.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: sinh(x)
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.cosh(x))

    def fwd(self, x, **kwargs):
        return jnp.sinh(x)

    def rev(self, y, **kwargs):
        return jnp.arcsinh(y)


class AffineLinear(ScalarBijection):
    """Learnable affine transformation.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: scale * x + shift
    """

    def __init__(
        self,
        scale: ParamSpec = (),
        shift: ParamSpec = (),
        transform_scale: Callable | None = jnp.exp,
        transform_shift: Callable | None = None,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.scale = TransformedParameter(
            param=default_wrap(scale, init_fn=nnx.initializers.zeros, rngs=rngs),
            transform=transform_scale,
        )
        self.shift = TransformedParameter(
            param=default_wrap(shift, init_fn=nnx.initializers.zeros, rngs=rngs),
            transform=transform_shift,
        )

    def log_jac(self, x, y, **kwargs):
        return jnp.broadcast_to(jnp.log(self.scale.value), jnp.shape(x))

    def fwd(self, x, **kwargs):
        return self.scale.value * x + self.shift.value

    def rev(self, y, **kwargs):
        return (y - self.shift.value) / self.scale.value


class Scaling(ScalarBijection):
    """Scaling transformation.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: scale * x
    """

    def __init__(
        self,
        scale: ParamSpec = (),
        transform_scale: Callable | None = None,
        *,
        rngs=None,
    ):
        self.scale = TransformedParameter(
            param=default_wrap(scale, init_fn=nnx.initializers.ones, rngs=rngs),
            transform=transform_scale,
        )

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(self.scale.value))

    def fwd(self, x, **kwargs):
        return x * self.scale.value

    def rev(self, y, **kwargs):
        return y / self.scale.value


class Shift(ScalarBijection):
    """Shift transformation.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: x + shift
    """

    def __init__(
        self,
        shift: ParamSpec = (),
        transform_shift: Callable | None = None,
        *,
        rngs=None,
    ):
        self.shift = TransformedParameter(
            param=default_wrap(shift, init_fn=nnx.initializers.zeros, rngs=rngs),
            transform=transform_shift,
        )

    def log_jac(self, x, y, **kwargs):
        return jnp.zeros_like(x)

    def fwd(self, x, **kwargs):
        return x + self.shift.value

    def rev(self, y, **kwargs):
        return y - self.shift.value


class BetaStretch(ScalarBijection):
    """Beta-inspired stretching on unit interval.

    Type: [0, 1] → [0, 1]
    Transform: x^a / (x^a + (1-x)^a).
    """

    def __init__(
        self,
        a: ParamSpec = (),
        transform_a: Callable | None = nnx.softplus,
        *,
        rngs=None,
    ):
        self.a = TransformedParameter(
            param=default_wrap(
                a, init_fn=nnx.initializers.constant(_softplus_inv_one), rngs=rngs
            ),
            transform=transform_a,
        )

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
    "AffineLinear",
    "BetaStretch",
    "Exponential",
    "GaussianCDF",
    "Power",
    "Scaling",
    "Shift",
    "Sigmoid",
    "Sinh",
    "Tan",
    "Tanh",
    "SoftPlus",
    "ScalarBijection",
]
