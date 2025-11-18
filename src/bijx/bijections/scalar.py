r"""
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
from flax import nnx

from ..utils import ParamSpec, ShapeInfo, default_wrap
from .base import Bijection


class TransformedParameter(nnx.Module):
    """Parameter wrapper with optional transformation function.

    Stores a raw parameter alongside an optional transformation function,
    providing a unified interface for accessing the transformed value.
    This conveniently facilitates constraints like positivity (e.g. exp/softplus)
    with unconstrained underlying parameters.

    Args:
        param: The underlying parameter variable.
        transform: Function to apply to param.value, or None for identity.

    Example:
        >>> param = nnx.Param(jnp.array(0.0))
        >>> transformed = TransformedParameter(param, jnp.exp)
        >>> transformed.value  # Returns exp(0.0) = 1.0
        Array(1., dtype=float64, weak_type=True)
    """

    def __init__(self, param: nnx.Variable, transform: Callable):
        self.param = param
        self.transform = transform

    @property
    def value(self):
        """Get the transformed parameter value."""
        transform = self.transform
        if transform is None:
            return self.param.value
        return transform(self.param.value)


_softplus_inv_one = jnp.log(jnp.expm1(1))


# not exported
def sum_log_jac(x, log_density, log_jac):
    """Sum log-Jacobian over event dimensions for density updates.

    Computes the updated log-density by summing the log-Jacobian contributions
    over the event dimensions while preserving batch dimensions. This function
    automatically infers event dimensions from the shape difference between
    input data and log-density arrays.

    Args:
        x: Input array with batch and event dimensions.
        log_density: Log density array with batch dimensions only.
        log_jac: Log-Jacobian contributions to sum over event dimensions.

    Returns:
        Updated log density with Jacobian contributions summed over events.
    """
    event_dim = jnp.ndim(x) - jnp.ndim(log_density)

    # Handle scalar case where there are no event dimensions to sum over
    if event_dim <= 0:
        return log_density + log_jac

    si = ShapeInfo(event_dim=event_dim, channel_dim=0)
    _, si = si.process_event(jnp.shape(x))
    return log_density + jnp.sum(log_jac, axis=si.event_axes)


class ScalarBijection(Bijection):
    r"""Base class for element-wise one-dimensional bijections.

    This abstract class provides the foundation for scalar (element-wise)
    bijections. It automatically handles log-density updates by
    summing log-Jacobian contributions over event dimensions, following the
    change of variables formula for element-wise transformations.

    Subclasses must implement:
        - :meth:`log_jac`: Log absolute determinant of Jacobian $\log \abs{f'(x)}$
        - :meth:`fwd`: Forward transformation $y = f(x)$
        - :meth:`rev`: Reverse transformation $x = f^{-1}(y)$

    The :meth:`forward` and :meth:`reverse` methods are implemented automatically
    and handle log-density updates by summing scalar log-Jacobians over event axes.

    Important:
        The ``forward`` and ``reverse`` methods should NOT be overridden.
    """

    def log_jac(self, x, y, **kwargs):
        r"""Compute log absolute determinant of the Jacobian.

        Args:
            x: Input values where Jacobian is computed.
            y: Output values corresponding to x (i.e., y = fwd(x)).
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Log absolute Jacobian determinant $\log \abs{f'(x)}$ with same shape as x.
        """
        raise NotImplementedError()

    def fwd(self, x, **kwargs):
        r"""Apply forward transformation.

        Args:
            x: Input values to transform.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Transformed values $y = f(x)$ with same shape as $x$.
        """
        raise NotImplementedError()

    def rev(self, y, **kwargs):
        r"""Apply reverse (inverse) transformation.

        Args:
            y: Output values to inverse-transform.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Inverse-transformed values $x = f^{-1}(y)$ with same shape as $y$.
        """
        raise NotImplementedError()

    def forward(self, x, log_density, **kwargs):
        """Apply forward transformation with log-density update.

        Transforms input through the bijection and updates log-density by
        subtracting the log-Jacobian determinant, summed over event dimensions.

        Args:
            x: Input data to transform.
            log_density: Log density values for the input.
            **kwargs: Additional arguments passed to :meth:`fwd` and :meth:`log_jac`.

        Returns:
            Tuple of (transformed_data, updated_log_density).
        """
        y = self.fwd(x, **kwargs)
        return y, sum_log_jac(x, log_density, -self.log_jac(x, y))

    def reverse(self, y, log_density, **kwargs):
        """Apply reverse transformation with log-density update.

        Transforms input through the inverse bijection and updates log-density by
        adding the log-Jacobian determinant, summed over event dimensions.

        Args:
            y: Input data to inverse-transform.
            log_density: Log density values for the input.
            **kwargs: Additional arguments passed to :meth:`rev` and :meth:`log_jac`.

        Returns:
            Tuple of (inverse_transformed_data, updated_log_density).
        """
        x = self.rev(y, **kwargs)
        return x, sum_log_jac(x, log_density, self.log_jac(x, y))


class GaussianCDF(ScalarBijection):
    r"""Bijection via Gaussian CDF with learnable location and scale.

    Transforms unbounded inputs to the unit interval using the Gaussian cumulative
    distribution function. The transformation first standardizes inputs using
    learnable location and scale parameters, then applies the standard normal CDF.

    Type: $[-\infty, \infty] \to [0, 1]$

    Transform: $\Phi\left(\frac{x - \mu}{\sigma}\right)$
    where $\Phi$ is the standard Gaussian CDF

    The log-Jacobian is the Gaussian log-PDF:
    $\log \abs{f'(x)} = \log \phi\left(\frac{x - \mu}{\sigma}\right) - \log \sigma$

    Args:
        scale: Scale parameter specification, transformed to ensure positivity.
        mean: Mean parameter specification, no transformation by default.
        transform_scale: Function to ensure positive scale (default: softplus).
        transform_mean: Function to transform mean (default: None/identity).
        rngs: Random number generators for parameter initialization.

    Note:
        By default, the scale parameter is initialized to give unit scale after
        transformation, and the mean parameter is initialized to zero.

    Example:
        >>> bijection = GaussianCDF(rngs=rngs)
        >>> x = jnp.array([-2.0, 0.0, 2.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
        >>> # y ≈ [0.023, 0.5, 0.977] (standard normal CDF values)
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
    r"""Tangent-based unbounded transform.

    Maps the unit interval to the real line using the tangent function.
    The transformation centers the input around 0.5, scales by π, then applies
    tangent to achieve the unbounded output domain.

    Type: $[0, 1] \to [-\infty, \infty]$

    Transform: $\tan(\pi(x - 0.5))$

    Example:
        >>> bijection = Tan()
        >>> x = jnp.array([0.25, 0.5, 0.75])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(jnp.pi * (1 + y**2)))

    def fwd(self, x, **kwargs):
        return jnp.tan(jnp.pi * (x - 0.5))

    def rev(self, y, **kwargs):
        return jnp.arctan(y) / jnp.pi + 0.5


class Sigmoid(ScalarBijection):
    r"""Sigmoid normalization transform.

    Maps the real line to the unit interval using the logistic sigmoid function.

    Type: $[-\infty, \infty] \to [0, 1]$

    Transform: $\sigma(x) = \frac{1}{1 + e^{-x}}$

    Example:
        >>> bijection = Sigmoid()
        >>> x = jnp.array([-2.0, 0.0, 2.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
    """

    def log_jac(self, x, y):
        return jnp.log(y) + jnp.log(1 - y)

    def fwd(self, x, **kwargs):
        return nnx.sigmoid(x)

    def rev(self, y, **kwargs):
        return jnp.log(y / (1 - y))


class Tanh(ScalarBijection):
    r"""Hyperbolic tangent bounded transform.

    Maps the real line to the interval $[-1, 1]$ using the hyperbolic tangent
    function, providing a symmetric bounded transformation.

    Type: $[-\infty, \infty] \to [-1, 1]$

    Transform: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

    Example:
        >>> bijection = Tanh()
        >>> x = jnp.array([-2.0, 0.0, 2.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.abs(1 - y**2))

    def fwd(self, x, **kwargs):
        return jnp.tanh(x)

    def rev(self, y, **kwargs):
        return jnp.arctanh(y)


class Exponential(ScalarBijection):
    r"""Exponential transform to positive reals.

    Maps the real line to positive reals using the exponential function.

    Type: $[-\infty, \infty] \to [0, \infty]$

    Transform: $\exp(x)$

    Example:
        >>> bijection = Exponential()
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
        >>> # y ≈ [0.368, 1.0, 2.718]
    """

    def log_jac(self, x, y, **kwargs):
        return x

    def fwd(self, x, **kwargs):
        return jnp.exp(x)

    def rev(self, y, **kwargs):
        return jnp.log(y)


class SoftPlus(ScalarBijection):
    r"""SoftPlus transform.

    Maps the real line to positive reals using the softplus function.

    Type: $[-\infty, \infty] \to [0, \infty]$

    Transform: $\text{softplus}(x) = \log(1 + e^x)$

    Example:
        >>> bijection = SoftPlus()
        >>> x = jnp.array([-5.0, 0.0, 5.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
    """

    def log_jac(self, x, y, **kwargs):
        return -nnx.softplus(-x)

    def fwd(self, x, **kwargs):
        return nnx.softplus(x)

    def rev(self, y, **kwargs):
        return jnp.log(-jnp.expm1(-y)) + y


class Power(ScalarBijection):
    r"""Power transformation for positive values.

    Applies a power transformation to positive inputs with a learnable exponent.

    Type: $[0, \infty] \to [0, \infty]$

    Transform: $x^p$ where $p > 0$

    Args:
        exponent: Exponent parameter specification, transformed to ensure positivity.
        transform_exponent: Function to ensure positive exponent (default: abs).
        rngs: Random number generators for parameter initialization.

    Note:
        The constraint $p > 0$ is not validated, but by default enforced
        by the ``transform_exponent`` function.
        Thus could also set ``exponent`` to a constant negative value
        or integer (wrapped in ``Const``, to avoid training) after setting
        ``transform_exponent=None``.

    Example:
        >>> bijection = Power(rngs=rngs)  # Starts with p=1
        >>> x = jnp.array([0.5, 1.0, 2.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
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
    r"""Hyperbolic sine transformation.

    Maps the real line to itself using the hyperbolic sine function.
    This provides a smooth, odd function that grows exponentially for large
    $\abs{x}$ while remaining approximately linear near zero.
    Becomes numerically unstable for large $\abs{x}$.

    Type: $[-\infty, \infty] \to [-\infty, \infty]$

    Transform: $\sinh(x) = \frac{e^x - e^{-x}}{2}$

    Example:
        >>> bijection = Sinh()
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
    """

    def log_jac(self, x, y, **kwargs):
        return jnp.log(jnp.cosh(x))

    def fwd(self, x, **kwargs):
        return jnp.sinh(x)

    def rev(self, y, **kwargs):
        return jnp.arcsinh(y)


class AffineLinear(ScalarBijection):
    r"""Learnable affine transformation.

    Applies a learnable affine transformation combining scaling and shifting.
    This is one of the most fundamental bijections, providing location-scale
    transformations commonly used in normalizing flows.

    Type: $[-\infty, \infty] \to [-\infty, \infty]$

    Transform: $ax + b$ where $a$ (scale) and $b$ (shift) are learnable parameters

    Args:
        scale: Scale parameter specification, transformed to ensure appropriate scaling.
        shift: Shift parameter specification, no transformation by default.
        transform_scale: Function to transform scale (default: exp for positivity).
        transform_shift: Function to transform shift (default: None/identity).
        rngs: Random number generators for parameter initialization.

    Example:
        >>> bijection = AffineLinear(rngs=rngs)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
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
    r"""Scaling transformation.

    Applies element-wise scaling with a learnable scale parameter.
    This is a simpler version of :class:`AffineLinear` without the shift term.

    Type: $[-\infty, \infty] \to [-\infty, \infty]$

    Transform: $ax$ where $a$ is a learnable scale parameter

    Args:
        scale: Scale parameter specification.
        transform_scale: Function to transform scale (default: None/identity).
        rngs: Random number generators for parameter initialization.

    Note:
        The scale parameter is initialized to 1 (identity scaling).
        No positivity constraint is applied by default.

    Example:
        >>> bijection = Scaling(rngs=rngs)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
        >>> # Initially y == x (scale=1)
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
        return jnp.broadcast_to(jnp.log(jnp.abs(self.scale.value)), jnp.shape(x))

    def fwd(self, x, **kwargs):
        return x * self.scale.value

    def rev(self, y, **kwargs):
        return y / self.scale.value


class Shift(ScalarBijection):
    r"""Shift transformation.

    Applies element-wise shifting with a learnable shift parameter.
    This is a simpler version of :class:`AffineLinear` without the scale term.

    Type: $[-\infty, \infty] \to [-\infty, \infty]$

    Transform: $x + b$ where $b$ is a learnable shift parameter

    Args:
        shift: Shift parameter specification.
        transform_shift: Function to transform shift (default: None/identity).
        rngs: Random number generators for parameter initialization.

    Example:
        >>> bijection = Shift(rngs=rngs)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
        >>> all(y == x)  # Initially shift = 0, so y == x
        True
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
    r"""Beta-inspired stretching on unit interval.

    Applies a nonlinear stretching transformation on the unit interval inspired
    by the Beta distribution. This bijection can compress or expand different
    regions of [0,1] depending on the parameter value.

    Type: $[0, 1] \to [0, 1]$

    Transform: $f(x) = \frac{x^a}{x^a + (1-x)^a}$ where $a > 0$

    Args:
        a: Shape parameter controlling the stretching behavior.
        transform_a: Function to ensure positive parameter (default: softplus).
        rngs: Random number generators for parameter initialization.

    Note:
        - When a = 1, this is the identity transformation
        - When a < 1, the transformation is S-shaped (sigmoidal)
        - When a > 1, the transformation is reverse S-shaped
        - The parameter is initialized to give a = 1 after transformation

    Example:
        >>> bijection = BetaStretch(rngs=rngs)
        >>> x = jnp.array([0.25, 0.5, 0.75])
        >>> y, log_det = bijection.forward(x, jnp.zeros(3))
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
