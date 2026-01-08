from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax_autovmap import autovmap

from ..utils import ParamSpec, default_wrap
from .scalar import ScalarBijection, TransformedParameter

_softplus_inv_one = jnp.log(jnp.expm1(1))


@dataclass
class SigmoidTransform:
    low: float = -1
    high: float = 8
    eps_low: float = 1e-3
    eps_high: float = 1e-3

    def __call__(self, a):
        low = self.low + self.eps_low
        high = self.high - self.eps_high
        diff = high - low
        offset = jax.scipy.special.logit(-low / diff)
        return low + diff * nnx.sigmoid(a + offset)


@dataclass
class SoftplusTransform:
    eps: float = 1e-1

    def __call__(self, b):
        return self.eps + nnx.softplus(b + 1)


@jax.jit
@autovmap(a=0, b=0, c=0, d=0)
def solve_cubic(a, b, c, d):
    """Solve cubic equation ax³ + bx² + cx + d = 0 using Cardano's formula.

    Uses numerically stable computation for the real root.
    """
    d0 = b**2 - 3 * a * c
    d1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    sqrt = jnp.sqrt(d1**2 - 4 * d0**3)

    minus = d1 - sqrt
    plus = d1 + sqrt
    c_arg = jnp.where(
        jnp.abs(minus) < jnp.abs(plus),
        plus,
        minus,
    )
    c = jnp.cbrt(c_arg / 2)
    return -(b + c + d0 / c) / (3 * a)


class CubicRational(ScalarBijection):
    """Modified rational transform with learnable parameters.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: x + α*x/(1 + β*x²) with constrained α ∈ [-1,8], β > 0.
    """

    def __init__(
        self,
        loc: ParamSpec = (),
        alpha: ParamSpec = (),
        beta: ParamSpec = (),
        alpha_transform: Callable | None = SigmoidTransform(),
        beta_transform: Callable | None = SoftplusTransform(),
        loc_transform: Callable | None = None,
        *,
        rngs=None,
    ):
        self.alpha = TransformedParameter(
            param=default_wrap(alpha, rngs=rngs, init_fn=nnx.initializers.normal()),
            transform=alpha_transform,
        )
        self.beta = TransformedParameter(
            param=default_wrap(beta, rngs=rngs, init_fn=nnx.initializers.normal(2)),
            transform=beta_transform,
        )
        self.loc = TransformedParameter(
            param=default_wrap(loc, rngs=rngs, init_fn=jax.random.normal),
            transform=loc_transform,
        )

    def log_jac(self, x, y):
        x = x - self.loc.get_value()
        bx1 = self.beta.get_value() * x**2 + 1
        return -jnp.log(bx1) + jnp.log(
            bx1
            + self.alpha.get_value()
            - 2 * self.alpha.get_value() * self.beta.get_value() * x**2 / bx1
        )

    def fwd(self, x, **kwargs):
        x = x - self.loc.get_value()
        y = x + self.alpha.get_value() * x / (1 + self.beta.get_value() * x**2)
        return y + self.loc.get_value()

    def rev(self, y, **kwargs):
        y = y - self.loc.get_value()
        x = solve_cubic(
            self.beta.get_value(),
            -self.beta.get_value() * y,
            self.alpha.get_value() + 1,
            -y,
        )
        return x + self.loc.get_value()


@jax.jit
@autovmap(x=0, beta=0, mu=0, nu=0)
def sinh_nonlinearity(x, beta=0.0, mu=0.0, nu=0.0, threshold=15.0):
    """Bijective nonlinearity: f(x) = arcsinh(exp(mu) * (exp(nu) * sinh(x) + beta)).

    Inverse is given by mu=-nu, nu=-mu, beta=-beta.

    Threshold is used for numerical stability.
    """
    # For |x| >= threshold, use asymptotic forms
    log_mu_nu = mu + nu

    # Middle range computation with protected sinh
    x_clamped = jnp.where(jnp.abs(x) < threshold, x, 0.0)
    sinh_val = jnp.sinh(x_clamped)
    arg = jnp.exp(mu) * (jnp.exp(nu) * sinh_val + beta)
    middle_result = jnp.arcsinh(arg)

    # Asymptotic approximations
    result = jnp.where(
        x >= threshold,
        x + log_mu_nu,  # Large positive: f(x) ≈ x + log(mu*nu)
        jnp.where(
            x <= -threshold,
            x - log_mu_nu,  # Large negative: f(x) ≈ x - log(mu*nu)
            middle_result,
        ),
    )

    return result


@jax.jit
def log_cosh_stable(x):
    """
    Stable computation of log(cosh(x)).

    Uses: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)

    For large |x|: log(cosh(x)) ≈ |x| - log(2)
    """
    abs_x = jnp.abs(x)
    # log(1 + exp(-2|x|)) vanishes for large |x|, so this is stable
    return abs_x + jnp.log1p(jnp.exp(-2.0 * abs_x)) - jnp.log(2.0)


@jax.jit
@autovmap(x=0, beta=0, mu=0, nu=0)
def log_grad_sinh_nonlinearity(x, beta=0.0, mu=1.0, nu=1.0, threshold=15.0):
    """
    Numerically stable computation of log(f'(x)) where
    f(x) = arcsinh(exp(mu) * (exp(nu) * sinh(x) + beta)).
    """
    # For large |x|, the gradient approaches 1, so log(gradient) → 0
    asymptotic_result = 0.0

    # Check if we're in asymptotic regime
    in_asymptotic = jnp.abs(x) > threshold

    # CRITICAL: Clamp x BEFORE any computation to ensure both branches are safe
    # This prevents sinh overflow even when gradients flow through the middle branch
    x_clamped = jnp.clip(x, -threshold, threshold)
    sinh_x = jnp.sinh(x_clamped)

    # Compute exp(mu) and exp(nu) - protect against overflow
    exp_mu_raw = jnp.exp(mu)
    exp_nu_raw = jnp.exp(nu)
    exp_max = 1e10
    exp_mu = jnp.where(
        jnp.isfinite(exp_mu_raw), jnp.clip(exp_mu_raw, 0.0, exp_max), exp_max
    )
    exp_nu = jnp.where(
        jnp.isfinite(exp_nu_raw), jnp.clip(exp_nu_raw, 0.0, exp_max), exp_max
    )

    # Compute arg - safe because sinh_x is bounded
    beta_safe = jnp.clip(beta, -1e5, 1e5)
    arg_raw = exp_mu * (exp_nu * sinh_x + beta_safe)

    # Protect arg from becoming inf/nan
    arg_max = 1e10
    arg = jnp.where(jnp.isfinite(arg_raw), jnp.clip(arg_raw, -arg_max, arg_max), 0.0)
    arg_squared = arg * arg

    # Stable log(cosh(x)) computation - safe because x_clamped is bounded
    log_cosh_x = log_cosh_stable(x_clamped)

    # Stable computation of log(sqrt(1 + arg^2)) = 0.5 * log(1 + arg^2)
    # For small arg^2, use log1p for accuracy
    # For large arg^2, use 0.5 * log(arg^2) which is more stable on GPU than log(|arg|)
    threshold_arg_sq = 1e8  # Lower threshold for smoother transition on GPU

    # When arg^2 is small: use log1p for precision
    log_sqrt_denom_small = 0.5 * jnp.log1p(arg_squared)

    # When arg^2 is large: log(sqrt(1 + arg^2)) ≈ 0.5 * log(arg^2) = log(|arg|)
    # Use 0.5 * log(arg^2) directly for better GPU numerical stability
    # Add small epsilon to prevent log(0) in edge cases
    log_sqrt_denom_large = 0.5 * jnp.log(jnp.maximum(arg_squared, 1e-20))

    log_sqrt_denom = jnp.where(
        arg_squared < threshold_arg_sq, log_sqrt_denom_small, log_sqrt_denom_large
    )

    # Final safeguard: ensure result is finite (handles GPU numerical edge cases)
    log_sqrt_denom = jnp.where(
        jnp.isfinite(log_sqrt_denom), log_sqrt_denom, log_sqrt_denom_small
    )

    # Combine - this is safe even when |x| > threshold because x_clamped is bounded
    middle_result = mu + nu + log_cosh_x - log_sqrt_denom

    # Select based on x magnitude - both branches are now safe
    result = jnp.where(in_asymptotic, asymptotic_result, middle_result)

    return result


class SinhConjugation(ScalarBijection):
    """Sinh-based bijection using conjugation with arcsinh.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: arcsinh(exp(mu) * (exp(nu) * sinh((x-loc)/alpha) + beta)) * alpha + loc

    Parameters:
        loc: Location parameter (shift)
        alpha: Scale parameter (must be positive)
        beta: Offset parameter in sinh space
        mu: Log-scale parameter for outer stretch
        nu: Log-scale parameter for inner stretch
    """

    def __init__(
        self,
        loc: ParamSpec = (),
        alpha: ParamSpec = (),
        beta: ParamSpec = (),
        mu: ParamSpec = (),
        nu: ParamSpec = (),
        alpha_transform: Callable | None = lambda x: nnx.softplus(x) + 0.01,
        mu_transform: Callable | None = jnp.arcsinh,
        nu_transform: Callable | None = jnp.arcsinh,
        rngs=None,
    ):
        self.alpha = TransformedParameter(
            param=default_wrap(alpha, rngs=rngs, init_fn=nnx.initializers.normal()),
            transform=alpha_transform,
        )
        self.loc = default_wrap(loc, rngs=rngs, init_fn=nnx.initializers.normal(1))
        self.beta = default_wrap(beta, rngs=rngs, init_fn=nnx.initializers.normal())
        self.mu = TransformedParameter(
            param=default_wrap(mu, rngs=rngs, init_fn=nnx.initializers.zeros_init()),
            transform=mu_transform,
        )
        self.nu = TransformedParameter(
            param=default_wrap(nu, rngs=rngs, init_fn=nnx.initializers.zeros_init()),
            transform=nu_transform,
        )

    def _params(self):
        beta = self.beta.get_value()
        loc = self.loc.get_value()
        alpha = self.alpha.get_value()
        mu = self.mu.get_value()
        nu = self.nu.get_value()
        return alpha, beta, loc, mu, nu

    def log_jac(self, x, y):
        alpha, beta, loc, mu, nu = self._params()
        a = (x - loc) / alpha
        return log_grad_sinh_nonlinearity(a, beta, mu, nu)

    def fwd(self, x, **kwargs):
        alpha, beta, loc, mu, nu = self._params()
        a = (x - loc) / alpha
        return sinh_nonlinearity(a, beta, mu, nu) * alpha + loc

    def rev(self, y, **kwargs):
        alpha, beta, loc, mu, nu = self._params()
        a = (y - loc) / alpha
        return sinh_nonlinearity(a, -beta, -nu, -mu) * alpha + loc


def _cubic_forward(x, a=1, b=1):
    return a * x + b * x**3


def _cubic_reverse(y, a=1, b=1):
    return solve_cubic(b, 0, a, -y)


@jax.jit
@autovmap(x=0, a=0, b=0, beta=0)
def cubic_nonlinearity(x, a=1, b=1, beta=0):
    return _cubic_reverse(_cubic_forward(x, a, b) + beta, a, b)


@jax.jit
@autovmap(x=0, a=0, b=0, beta=0)
def log_grad_cubic_nonlinearity(x, a=1, b=1, beta=0):
    grad = jax.grad(cubic_nonlinearity, argnums=0)(x, a, b, beta)
    return jnp.log(jnp.abs(grad) + 1e-12)


class CubicConjugation(ScalarBijection):
    """Cubic polynomial-based bijection.

    Type: [-∞, ∞] → [-∞, ∞]
    Transform: Based on cubic polynomial a*x + b*x³ with conjugation offset

    Parameters:
        loc: Location parameter (shift)
        beta: Offset parameter for conjugation
        a: Linear coefficient (must be positive)
        b: Cubic coefficient (must be positive)
    """

    def __init__(
        self,
        loc: ParamSpec = (),
        beta: ParamSpec = (),
        a: ParamSpec = (),
        b: ParamSpec = (),
        a_transform: Callable | None = nnx.softplus,
        b_transform: Callable | None = nnx.softplus,
        rngs=None,
    ):
        self.loc = default_wrap(loc, rngs=rngs, init_fn=nnx.initializers.normal(1))
        self.beta = default_wrap(beta, rngs=rngs, init_fn=nnx.initializers.normal())
        self.a = TransformedParameter(
            param=default_wrap(a, rngs=rngs, init_fn=nnx.initializers.normal(1)),
            transform=a_transform,
        )
        self.b = TransformedParameter(
            param=default_wrap(b, rngs=rngs, init_fn=nnx.initializers.normal(1)),
            transform=b_transform,
        )

    def _params(self):
        beta = self.beta.get_value()
        loc = self.loc.get_value()
        a = self.a.get_value()
        b = self.b.get_value()
        return beta, loc, a, b

    def log_jac(self, x, y):
        beta, loc, a, b = self._params()
        return log_grad_cubic_nonlinearity(x - loc, a, b, beta)

    def fwd(self, x, **kwargs):
        beta, loc, a, b = self._params()
        return cubic_nonlinearity(x - loc, a, b, beta) + loc

    def rev(self, y, **kwargs):
        beta, loc, a, b = self._params()
        # Inverse: reverse the conjugation by swapping sign of beta
        return cubic_nonlinearity(y - loc, a, b, -beta) + loc
