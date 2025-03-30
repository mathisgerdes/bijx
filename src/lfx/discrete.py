from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .bijections import Bijection
from .utils import Const


def checker_mask(shape, parity: bool):
    """Checkerboard mask.

    Args:
        shape: Spacial dimensions of input.
        parity: Parity of mask.
    """
    idx_shape = np.ones_like(shape)
    idc = []
    for i, s in enumerate(shape):
        idx_shape[i] = s
        idc.append(np.arange(s, dtype=np.uint8).reshape(idx_shape))
        idx_shape[i] = 1
    mask = (sum(idc) + parity) % 2
    return mask


class AffineCoupling(Bijection):
    """
    Affine coupling layer.

    Example:
    ```python
    space_shape = (16, 16)  # no channel/feature dim (add dummy axis below)

    affine_flow = lfx.Chain([
        lfx.ExpandDims(),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, 2, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), True)),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, 2, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), False)),
        lfx.ExpandDims().invert(),
    ])
    ```

    `net` should map: `x_f -> act`
    such that `s, t = split(act, 2, -1)`
    and `x_out = t + x_a * exp(s) + x_f`

    Args:
        net: Network that maps frozen features to s, t.
        mask: Mask to apply to input.
    """

    def __init__(self, net: nnx.Module, mask: jax.Array, *, rngs=None):
        self.mask = Const(mask)
        self.net = net

    @property
    def mask_active(self):
        return 1 - self.mask.value

    @property
    def mask_frozen(self):
        return self.mask.value

    def forward(self, x, log_density):
        x_frozen = self.mask_frozen * x
        x_active = self.mask_active * x
        activation = self.net(x_frozen)
        s, t = jnp.split(activation, 2, -1)
        fx = x_frozen + self.mask_active * t + x_active * jnp.exp(s)
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return fx, log_density - log_jac

    def reverse(self, fx, log_density):
        fx_frozen = self.mask_frozen * fx
        fx_active = self.mask_active * fx
        activation = self.net(fx_frozen)
        s, t = jnp.split(activation, 2, -1)
        x = (fx_active - self.mask_active * t) * jnp.exp(-s) + fx_frozen
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return x, log_density + log_jac


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

    sline_params = jax.random.normal(rngs(), (15, x_dim * spline.spline_param_count))
    y, log_prob = spline.forward(x, log_prob, spline_params=sline_params)
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
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.params_net = params_net

    @property
    def spline_param_count(self):
        return 3 * self.knots - 1

    @property
    def spline_param_splits(self):
        return (self.knots, 2 * self.knots)

    def __call__(self, x, inverse=False, **kwargs):
        params = self.params_net(**kwargs)

        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        params = params.reshape(x.shape[0], self.spline_param_count * x.shape[1])

        w, h, d = jnp.split(
            params,
            self.spline_param_splits,
            axis=-1,
        )

        x, delta_log_density = apply_mrq_spline(
            x,
            w,
            h,
            d,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        # add this to log_density
        delta_log_density = -jnp.sum(delta_log_density, axis=-1)

        return x.reshape(shape), delta_log_density.reshape(shape[:-1])

    def forward(self, x, log_density, **kwargs):
        x, delta_log_density = self(x, **kwargs)
        return x, log_density + delta_log_density

    def reverse(self, x, log_density, **kwargs):
        x, delta_log_density = self(x, inverse=True, **kwargs)
        return x, log_density + delta_log_density
