import jax
import jax.numpy as jnp
from flax import nnx

from .bijections import Bijection
from .utils import default_wrap


class GaussianCDF(Bijection):
    """Invertible map from [-inf, inf] to [0, 1] using Gaussian cdfs."""

    def __init__(
        self,
        init_log_scale: jax.Array | nnx.Variable = jnp.zeros(()),
        init_mean: jax.Array | nnx.Variable = jnp.zeros(()),
        *,
        rngs: nnx.Rngs = None,
    ):
        self.mean = default_wrap(init_mean)
        self.log_scale = default_wrap(init_log_scale)

    def forward(self, x, logp):
        loc = self.mean.value
        scale = jnp.exp(self.log_scale)
        y = jax.scipy.stats.norm.cdf(x, loc=loc, scale=scale)
        log_jac = jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale)

        return y, logp - log_jac

    def reverse(self, y, logp):
        loc = self.mean.value
        scale = jnp.exp(self.log_scale)
        x = jax.scipy.stats.norm.ppf(y, loc=loc, scale=scale)
        log_jac = jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale)

        return x, logp + log_jac


class TanLayer(Bijection):
    """Invertible map from [0, 1] to [-inf, inf] using tan/arctan."""

    def forward(self, x, log_density):
        y = jnp.tan(jnp.pi * (x + 0.5))
        jac = jnp.abs(jnp.pi * (1 + y**2))
        return y, log_density - jnp.log(jac)

    def reverse(self, y, log_density):
        x = jnp.arctan(y) / jnp.pi + 0.5
        jac = jnp.abs(jnp.pi * (1 + y**2))
        return x, log_density + jnp.log(jac)


def _sigmoid_inv(x):
    return jnp.log(x / (1 - x))


class SigmoidLayer(Bijection):
    """Invertible map from [-inf, inf] to [0, 1] using sigmoid."""

    def forward(self, x, log_density):
        y = nnx.sigmoid(x)
        log_jac = jnp.log(y) + jnp.log(1 - y)
        return y, log_density - log_jac

    def reverse(self, y, log_density):
        x = _sigmoid_inv(y)
        log_jac = jnp.log(y) + jnp.log(1 - y)
        return x, log_density + log_jac


class TanhLayer(Bijection):
    """Invertible map from [-inf, inf] to [-1, 1] using tanh."""

    def forward(self, x, log_density):
        y = jnp.tanh(x)
        jac = jnp.abs(1 - y**2)
        return y, log_density - jnp.log(jac)

    def reverse(self, y, log_density):
        x = jnp.arctanh(y)
        jac = jnp.abs(1 - y**2)
        return x, log_density + jnp.log(jac)


class BetaStretch(Bijection):
    """Invertible map [0, 1] -> [0, 1] using beta function."""

    def __init__(self, a: nnx.Variable, *, rngs=None):
        self.a = a

    def _log_jac(self, x):
        a = self.a.value
        log_jac = (
            jnp.log(a)
            + jnp.log(x ** (self.a - 1) * (1 - x) ** a + x**a * (1 - x) ** (a - 1))
            - 2 * jnp.log(x**a + (1 - x) ** a)
        )
        return log_jac

    def forward(self, x, log_density):
        xa = x**self.a
        y = xa / (xa + (1 - x) ** self.a)
        return y, log_density - self._log_jac(x)

    def reverse(self, y, log_density):
        r = (y / (1 - y)) ** (1 / self.a.value)
        x = r / (r + 1)
        return x, log_density + self._log_jac(x)
