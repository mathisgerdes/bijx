"""
Spline-based bijections.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import auto_vmap

from .base import Bijection


def apply_mrq_spline(
    x,
    w,
    h,
    d,
    *,
    inverse=False,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    # following arxiv: [1906.04032]
    # assumption: x.shape = (n,), others are (n, *)
    knots = w.shape[-1]

    w = nnx.softmax(w)
    h = nnx.softmax(h)

    w = min_bin_width + (1 - min_bin_width * knots) * w
    h = min_bin_height + (1 - min_bin_height * knots) * h

    # knots
    xs = jnp.pad(jnp.cumsum(w, -1), [(0, 0)] * (w.ndim - 1) + [(1, 0)])
    ys = jnp.pad(jnp.cumsum(h, -1), [(0, 0)] * (h.ndim - 1) + [(1, 0)])

    # derivatives
    beta = np.log(2) / (1 - min_derivative)
    deltas = min_derivative + nnx.softplus(beta * d) / beta

    @jax.vmap
    def _index(a, k):
        return a[k]

    if inverse:
        k = jax.vmap(partial(jnp.searchsorted, side="right"))(ys, x) - 1
    else:
        k = jax.vmap(partial(jnp.searchsorted, side="right"))(xs, x) - 1

    x_k = _index(xs, k)
    x_diff = _index(w, k)
    y_k = _index(ys, k)
    d_k = _index(deltas, k)
    y_diff = _index(h, k)
    d_k1 = _index(deltas, k + 1)

    s_k = y_diff / x_diff

    if inverse:
        a = (x - y_k) * (d_k + d_k1 - 2 * s_k) + y_diff * (s_k - d_k)
        b = y_diff * d_k - (x - y_k) * (d_k + d_k1 - 2 * s_k)
        c = -s_k * (x - y_k)

        discriminant = b**2 - 4 * a * c

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * x_diff + x_k

        # density
        r1r = root * (1 - root)
        denominator = s_k + ((d_k + d_k1 - 2 * s_k) * r1r)
        derivative_numerator = s_k**2 * (
            d_k1 * root**2 + 2 * s_k * r1r + d_k * (1 - root) ** 2
        )
        log_det = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)

        return outputs, -log_det

    xi = (x - x_k) / x_diff

    alpha = s_k * xi**2 + d_k * xi * (1 - xi)
    beta = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)

    out = y_k + y_diff * alpha / beta

    derivative_numerator = s_k**2 * (
        d_k1 * xi**2 + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2
    )
    log_det = jnp.log(derivative_numerator) - 2 * jnp.log(beta)
    return out, log_det


class MonotoneRQSpline(Bijection):
    """
    Monotone rational quadratic spline.

    Assumptions:
    - x (input) is 1-dimensional.
    - parameters returned by `params_net` has the same batch dimensions as x;
      both vmap'ed together.

    Example:
    ```python
    knots = 10
    x_dim = 5  # assume 1-dimensional for this class!

    spline = MonotoneRQSpline(
        knots=knots,
        # dummy network (in reality probably some network)
        params_net=lambda spline_params: spline_params
    )

    x  = jax.random.uniform(rngs(), (15, x_dim))
    log_prob = jnp.zeros((15,))
    ```
    """

    def __init__(
        self,
        knots: int,
        params_net: nnx.Module,
        *,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        rngs: nnx.Rngs | None = None,
    ):
        self.knots = knots
        self.params_net = params_net
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    @property
    def spline_param_count(self):
        return 3 * self.knots - 1

    @property
    def spline_param_splits(self):
        return [self.knots, self.knots, self.knots - 1]

    @auto_vmap(x=1, params=1)
    def __call__(self, x, params, inverse=False):
        w, h, d = jnp.split(params, self.spline_param_splits, axis=-1)
        return apply_mrq_spline(
            x,
            w,
            h,
            d,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

    def forward(self, x, log_density, **kwargs):
        params = self.params_net(x)
        x, log_jac = self(x, params, inverse=False)
        return x, log_density - log_jac

    def reverse(self, x, log_density, **kwargs):
        params = self.params_net(x)
        x, log_jac = self(x, params, inverse=True)
        return x, log_density + log_jac
