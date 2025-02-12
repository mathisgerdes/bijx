from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .bijections import Bijection, Const
from .sampling import Prior
from .utils import ShapeInfo


def fft_momenta(shape: tuple[int, ...], reduced: bool = True, lattice: bool = False) -> jax.Array:
    shape_factor = np.reshape(shape, [-1] + [1] * len(shape))
    if reduced:
        # using reality condition, can eliminate about half of components
        shape = list(shape)[:-1] + [np.floor(shape[-1] / 2) + 1]

    # get frequencies divided by shape as large grid
    # ks[i] is k varying along axis i from 0 to L_i
    ks = np.mgrid[tuple(np.s_[:s] for s in shape)] / shape_factor
    if lattice:
        # with this true, spectrum ~ 1 / m^2 + k^2
        # otherwise get ~ 1 / k^2 - 2 (cos(2 pi k) - 1)
        ks = 2 * jnp.sin(ks * jnp.pi)
    # move "i" to last axis
    return np.moveaxis(ks, 0, -1)


class SpectrumScaling(Bijection):
    """Real Gaussian diagonal in Fourier space.

    Note: scaling should be array of same shape as output of rfftn!
    """

    def __init__(self, scaling: jax.Array | nnx.Variable, channel_dim: int = 0):
        self.channel_dim = channel_dim

        if isinstance(scaling, (jax.Array, np.ndarray)):
            scaling = Const(scaling)
        self.scaling = scaling
        self.shape_info = ShapeInfo(space_shape=scaling.shape, channel_dim=channel_dim)

    @property
    def scaling_array(self):
        try:
            return self.scaling.value
        except AttributeError:
            return self.scaling

    def scale(self, r, reverse=False):
        _, shape_info = self.shape_info.process_event(r.shape)
        scaling = self.scaling_array

        r = jnp.fft.rfftn(r, shape_info.space_shape, shape_info.space_axes)
        r = r / scaling if reverse else r * scaling
        r = jnp.fft.irfftn(r, shape_info.space_shape, shape_info.space_axes)
        return r

    def forward(self, x, log_density):
        log_density = log_density + jnp.sum(jnp.log(self.scaling_array))
        return self.scale(x, reverse=False), log_density

    def reverse(self, x, log_density):
        log_density = log_density - jnp.sum(jnp.log(self.scaling_array))
        return self.scale(x, reverse=True), log_density


def get_fourier_masks(real_shape):
    """Get masks for independent d.o.f. of real FFT transform."""
    # rfft reduces last dimension
    rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

    real_mask = np.ones(rfft_shape, dtype=bool)
    imag_mask = np.ones(rfft_shape, dtype=bool)

    # right edge only if dimension is even length
    edges = [[0] + ([s//2] if s % 2 == 0 else [])
             for s in real_shape]

    cp_start = [s // 2 + 1 for s in real_shape[:-1]]

    # zero because reality condition makes value real
    for index in product(*edges):
        imag_mask[index] = False

    # zero because degrees of freedom are duplicated
    for i, s in enumerate(cp_start):
        for e1 in product(*edges[:i]):
            for e2 in product(*edges[i+1:]):
                real_mask[e1 + (np.s_[s:],) + e2] = False
                imag_mask[e1 + (np.s_[s:],) + e2] = False

    return real_mask, imag_mask
