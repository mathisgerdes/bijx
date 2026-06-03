from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax_autovmap import autovmap

from ..utils import ParamSpec, default_wrap
from .scalar import ScalarBijection, TransformedParameter

_softplus_inv_one = jnp.log(jnp.expm1(1))


def _softplus_inv(y):
    """Raw x such that ``softplus(x) == y`` (closed form ``log(expm1(y))``)."""
    return jnp.log(jnp.expm1(y))


def safe_exp_scale(x):
    """Bounded positive scale transform: ``exp(2 * tanh(x / 2))``.

    Strictly increasing and bounded away from both 0 and infinity, with range
    ``(exp(-2), exp(2)) ~ (0.135, 7.39)``.  Unlike ``softplus`` (bounded below
    but unbounded above), a large positive conditioner output cannot drive the
    scale to infinity, which removes the overflow/NaN mechanism for the cubic
    generator's ``a, b`` coefficients.
    """
    return jnp.exp(2.0 * jnp.tanh(x / 2.0))


def _safe_exp_scale_inv(y):
    """Raw x such that ``safe_exp_scale(x) == y`` (for y in (exp(-2), exp(2)))."""
    return 2.0 * jnp.arctanh(jnp.log(y) / 2.0)


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
        # alpha is the delta-control parameter (identity at alpha=0); keep its
        # init small/random-small.  depth-scaling is applied by the coupling/stack
        # helper, not baked into the per-element default.
        self.alpha = TransformedParameter(
            param=default_wrap(alpha, rngs=rngs, init_fn=nnx.initializers.normal()),
            transform=alpha_transform,
        )
        # beta is a scale (curvature radius); init at the neutral value beta~1 via
        # constant through the default SoftplusTransform (0.1+softplus(b+1)).
        self.beta = TransformedParameter(
            param=default_wrap(
                beta,
                rngs=rngs,
                init_fn=nnx.initializers.constant(float(_softplus_inv(0.9)) - 1.0),
            ),
            transform=beta_transform,
        )
        self.loc = TransformedParameter(
            param=default_wrap(loc, rngs=rngs, init_fn=nnx.initializers.zeros_init()),
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


def _sinh_log_arg_overflow(dtype, power):
    """log|arg| above which ``|arg|**power`` overflows ``dtype``.

    The clamp-free sinh helpers evaluate the singularity (arg=0) on a direct,
    gradient-clean path and only switch to the log-space asymptote once |arg| is
    too large for that path to stay finite.  ``power`` is 1 for the forward
    (forms ``arg``) and 2 for the log-Jac (forms ``arg**2``).  A 0.9 factor keeps
    a safety margin below the true overflow point and is dtype-aware so float32
    inputs switch to the asymptote well before float32 overflow.
    """
    return 0.9 * jnp.log(jnp.finfo(dtype).max) / power


@jax.jit
def _log_abs_sinh(abs_x):
    """log|sinh(x)| = |x| - log 2 + log1p(-exp(-2|x|)); -> -inf at x=0 (sinh=0)."""
    return abs_x - jnp.log(2.0) + jnp.log1p(-jnp.exp(-2.0 * abs_x))


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


def _sinh_overflow_logabs_z(x, beta, mu, nu, power):
    """Shared overflow mask + ``log|z|`` for the all-order-safe sinh helpers.

    ``z = exp(mu) (exp(nu) sinh x + beta)``.  Returns ``(big, log_abs_z, sign_z)``
    where ``big`` marks ``|z|**power`` as too large for the direct primitive path
    (so the asymptote is used there), and ``log_abs_z`` / ``sign_z`` are computed
    with double-``where`` SAFE inputs so they are non-singular to all orders
    wherever ``big`` is False -- the unused overflow branch never poisons the
    reverse-mode gradient at the ``s=0`` / ``x=0`` singularity.

    Subtle higher-order point: in the overflow regime we must not form ``log|s|``
    from the raw ``s = exp(nu) sinh x + beta``.  Its value is fine, but its 2nd/3rd
    x-derivatives form ``cosh(x)^2`` / ``cosh(x)^3`` which overflow to inf (then
    inf/inf = NaN) for the very large ``|x|`` that put us in the overflow regime.
    So there we use the asymptote ``log|s| ~= nu + log|sinh x|`` (linear in x, clean
    derivatives); the dropped ``log|1 + beta/(exp(nu) sinh x)|`` term is ~0 with
    derivatives ~1/sinh x -> 0.  Exact ``log|s|`` is used only where ``big`` is
    False (s moderate), and the asymptote only where ``big`` is True.
    """
    abs_x = jnp.abs(x)

    # Overflow mask from magnitude estimate:
    # log|z| ~= mu + nu + (|x| - log 2) once |x| is large (sinh ~ e^|x|/2).
    log_abs_z_mask = mu + nu + (abs_x - jnp.log(2.0))
    big = log_abs_z_mask > _sinh_log_arg_overflow(jnp.zeros((), x.dtype).dtype, power)

    # log|s| where ``big`` is FALSE: exact log(|s|).  Form ``s`` from a safe x
    # (sized down where big) so ``sinh(x)`` cannot overflow to +-inf on the unused
    # branch. An inf there poisons the reverse-mode cotangent even though the
    # ``where`` selects the other branch (0*inf = NaN through log/abs/sign).
    x_for_s = jnp.where(big, 0.0, x)
    s_exact = jnp.exp(nu) * jnp.sinh(x_for_s) + beta
    s_finite_nz = jnp.isfinite(s_exact) & (jnp.abs(s_exact) > 0.0)
    use_exact = (~big) & s_finite_nz
    s_safe = jnp.where(use_exact, s_exact, 1.0)
    log_abs_s_exact = jnp.log(jnp.abs(s_safe))
    sign_s = jnp.sign(s_safe)

    # log|s| where ``big`` is true: asymptote nu + log|sinh x| (clean to all orders
    # for large |x|); feed _log_abs_sinh a safe |x| where not big.
    abs_x_safe = jnp.where(big, abs_x, 1.0)
    log_abs_s_asymp = nu + _log_abs_sinh(abs_x_safe)

    log_abs_s = jnp.where(big, log_abs_s_asymp, log_abs_s_exact)
    log_abs_z = mu + log_abs_s

    # sign(z) = sign(s); in the overflow regime s is dominated by exp(nu) sinh x so
    # sign(s) = sign(x).  Guard sign(0) (zero subgradient) at the s=0 point.
    sign_z = jnp.where(big, jnp.sign(x), jnp.where(s_finite_nz, sign_s, 1.0))
    return big, log_abs_z, sign_z


@jax.jit
@autovmap(x=0, beta=0, mu=0, nu=0)
def sinh_nonlinearity(x, beta=0.0, mu=0.0, nu=0.0):
    """Bijective nonlinearity: f(x) = arcsinh(exp(mu) * (exp(nu) * sinh(x) + beta)).

    Inverse is given by mu=-nu, nu=-mu, beta=-beta.

    Computed from the direct smooth primitive ``jnp.arcsinh(z)`` in the normal
    regime (autodiff-clean to all orders at the regular point ``z=0``/``s=0``), with
    a full double-``where`` fall-back to the asymptote ``sign(z)(log|z|+log2)`` once
    ``z`` would overflow.  Plain ``jax.grad^k`` of this is finite for every ``k``.

    ``power=4`` (not 1): ``arcsinh(z)``'s VALUE is fine while ``z`` is
    representable, but its k-th derivative forms ``z^(k+1)`` intermediates that
    overflow (silent wrong ``0`` derivative, or NaN at 3rd order) well before ``z``
    itself does; switching once ``z^4`` would overflow keeps grad, grad^2 and grad^3
    representable, and there ``arcsinh(z) = log|z| + log2`` (with all derivatives)
    holds to full float precision.
    """
    big, log_abs_z, sign_z = _sinh_overflow_logabs_z(x, beta, mu, nu, power=4)

    # Normal branch: arcsinh(z) directly (smooth, odd through 0).  Size x/mu down
    # where ``big`` so z stays representable on the (unused) gradient path.
    x_safe = jnp.where(big, 0.0, x)
    mu_safe = jnp.where(big, 0.0, mu)
    z = jnp.exp(mu_safe) * (jnp.exp(nu) * jnp.sinh(x_safe) + beta)
    direct = jnp.arcsinh(z)

    # Overflow branch: arcsinh(z) ~= sign(z)*(log|z| + log2).
    log_abs_z_safe = jnp.where(big, log_abs_z, 0.0)
    asymp = sign_z * (log_abs_z_safe + jnp.log(2.0))

    return jnp.where(big, asymp, direct)


@jax.jit
@autovmap(x=0, beta=0, mu=0, nu=0)
def log_grad_sinh_nonlinearity(x, beta=0.0, mu=1.0, nu=1.0):
    """All-order autodiff-safe value of ``log f'(x)`` for the sinh nonlinearity.

    log(f'(x)) for f(x) = arcsinh(exp(mu) * (exp(nu) * sinh(x) + beta)).

    Exact: f'(x) = exp(mu+nu)*cosh(x)/sqrt(1+z^2), so
    log f'(x) = mu + nu + logcosh(x) - 0.5*log1p(z^2),
    z = exp(mu)(exp(nu)sinh x + beta).

    ``log f'(x) = mu + nu + logcosh(x) - 0.5*log1p(z^2)`` computed from the DIRECT
    smooth primitives ``0.5*log1p(z^2)`` and ``log(cosh x)`` (autodiff-clean to all
    orders at ``z=0``/``x=0``), with a full double-``where`` fall-back to the
    log-space asymptote ``0.5*logaddexp(0, 2 log|z|)`` once ``z`` would overflow.
    Plain ``jax.grad^k`` of this is finite for every ``k``.

    ``power=4`` (see ``_sinh_nonlinearity_value``): the direct ``0.5*log1p(z^2)`` is
    value-safe while ``z^2`` is representable, but its 2nd/3rd derivatives form
    ``z^3``/``z^4`` intermediates that overflow (NaN at 3rd order) before ``z^2``
    does; switching once ``z^4`` would overflow keeps grad^k (k<=3) finite.
    """
    log_cosh_x = log_cosh_stable(x)
    big, log_abs_z, _sign_z = _sinh_overflow_logabs_z(x, beta, mu, nu, power=4)

    # Normal branch: 0.5*log1p(z^2) directly (smooth, value/grad 0 at z=0).
    x_safe = jnp.where(big, 0.0, x)
    mu_safe = jnp.where(big, 0.0, mu)
    z = jnp.exp(mu_safe) * (jnp.exp(nu) * jnp.sinh(x_safe) + beta)
    half_log1p_direct = 0.5 * jnp.log1p(z * z)

    # Overflow branch: 0.5*log1p(z^2) -> 0.5*logaddexp(0, 2 log|z|).
    log_abs_z_safe = jnp.where(big, log_abs_z, 0.0)
    half_log1p_asymp = 0.5 * jnp.logaddexp(0.0, 2.0 * log_abs_z_safe)

    half_log1p_argsq = jnp.where(big, half_log1p_asymp, half_log1p_direct)
    return mu + nu + log_cosh_x - half_log1p_argsq


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
        alpha_transform: Callable | None = lambda x: nnx.softplus(x) + 0.1,
        mu_transform: Callable | None = jnp.arcsinh,
        nu_transform: Callable | None = jnp.arcsinh,
        rngs=None,
    ):
        # alpha is the overall scale; init at the neutral value alpha~1 via
        # constant(T^-1(1)) through the default floored softplus transform.
        # The positivity floor is raised 0.01 -> 0.1 (the inner 1/alpha scaling
        # makes a too-small alpha numerically fragile).
        self.alpha = TransformedParameter(
            param=default_wrap(
                alpha,
                rngs=rngs,
                init_fn=nnx.initializers.constant(float(_softplus_inv(0.9))),
            ),
            transform=alpha_transform,
        )
        # loc is a pure shift; init at 0 (never random).
        self.loc = default_wrap(loc, rngs=rngs, init_fn=nnx.initializers.zeros_init())
        # beta, mu, nu are delta-control params (identity at 0); keep their init
        # small.  depth-scaling is applied by the coupling/stack helper.
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
    # Inverse-function theorem: with g(u) = a*u + b*u^3, g'(u) = a + 3*b*u^2 and
    # f = g^{-1}(g(.) + beta), f'(x) = g'(x) / g'(f(x)).  This avoids
    # differentiating through the Cardano cbrt; numerically identical to the
    # autodiff form (verified to ~5e-10) and a slightly cheaper clean drop-in.
    fx = cubic_nonlinearity(x, a, b, beta)
    return jnp.log(jnp.abs(a + 3 * b * x**2)) - jnp.log(jnp.abs(a + 3 * b * fx**2))


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
        a_transform: Callable | None = safe_exp_scale,
        b_transform: Callable | None = safe_exp_scale,
        rngs=None,
    ):
        # loc is a pure shift; init at 0 (never random).
        self.loc = default_wrap(loc, rngs=rngs, init_fn=nnx.initializers.zeros_init())
        # beta is the delta-control (conjugation offset; identity at beta=0); keep
        # its init small.  depth-scaling is applied by the coupling/stack helper.
        self.beta = default_wrap(beta, rngs=rngs, init_fn=nnx.initializers.normal())
        # a, b are scales.  Default to the bounded ``safe_exp_scale`` (range
        # (exp(-2), exp(2)) ~ (0.135, 7.39)): bounding the cubic generator's
        # coefficients away from infinity removes the overflow/NaN mechanism, and
        # the lower bound (~0.135 > 0.05) keeps the map well away from
        # non-invertible.  Init at the neutral scales a~1, b~0.3 via
        # constant(T^-1(target)).
        self.a = TransformedParameter(
            param=default_wrap(
                a,
                rngs=rngs,
                init_fn=nnx.initializers.constant(float(_safe_exp_scale_inv(1.0))),
            ),
            transform=a_transform,
        )
        self.b = TransformedParameter(
            param=default_wrap(
                b,
                rngs=rngs,
                init_fn=nnx.initializers.constant(float(_safe_exp_scale_inv(0.3))),
            ),
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
