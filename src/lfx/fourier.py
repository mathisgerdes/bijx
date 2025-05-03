from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .bijections import Bijection
from .utils import Const, ShapeInfo


def fft_momenta(
    shape: tuple[int, ...], reduced: bool = True, lattice: bool = False
) -> jax.Array:
    shape_factor = np.reshape(shape, [-1] + [1] * len(shape))
    if reduced:
        # using reality condition, can eliminate about half of components
        shape = list(shape)[:-1] + [np.floor(shape[-1] / 2) + 1]

    # get frequencies divided by shape as large grid
    # ks[i] is k varying along axis i from 0 to L_i
    ks = np.mgrid[tuple(np.s_[:s] for s in shape)]
    ks = 2 * jnp.pi * ks / shape_factor
    if lattice:
        # with this true, (finite) lattice spectrum ~ 1 / m^2 + k^2
        # otherwise get ~ 1 / k^2 - 2 (cos(2 pi k) - 1)
        ks = 2 * jnp.sin(ks / 2)
    # move "i" (space-dim index) to last axis
    return np.moveaxis(ks, 0, -1)


class SpectrumScaling(Bijection):
    """Real Gaussian diagonal in Fourier space.

    Note: scaling should be array of same shape as output of rfftn!
    """

    def __init__(self, scaling: jax.Array | nnx.Variable, channel_dim: int = 0):
        self.channel_dim = channel_dim

        if not isinstance(scaling, nnx.Variable):
            scaling = Const(scaling)
        self.scaling_var = scaling
        self.shape_info = ShapeInfo(
            space_dim=len(scaling.shape), channel_dim=channel_dim
        )

    @property
    def scaling(self):
        return self.scaling_var.value

    def scale(self, r, reverse=False):
        _, shape_info = self.shape_info.process_event(r.shape)

        r = jnp.fft.rfftn(r, shape_info.space_shape, shape_info.space_axes)
        r = r / self.scaling if reverse else r * self.scaling
        r = jnp.fft.irfftn(r, shape_info.space_shape, shape_info.space_axes)

        mr, mi = get_fourier_masks(shape_info.space_shape)
        factor = mr.astype(int) + mi.astype(int)
        delta_ld = jnp.sum(factor * jnp.log(jnp.abs(self.scaling)))

        return r, delta_ld

    def forward(self, x, log_density):
        x, delta_ld = self.scale(x, reverse=False)
        return x, log_density - delta_ld

    def reverse(self, x, log_density):
        x, delta_ld = self.scale(x, reverse=True)
        return x, log_density + delta_ld


class FreeTheoryScaling(SpectrumScaling):
    """Scaling bijection which scales normal samples to free theory spectrum.

    Attributes:
        m2: The mass squared parameter. Can be a callable or a scalar value.
        space_shape: The shape of the event.
        channel_shape: The shape of the channel.
        finite_size: Whether to consider finite size effects.
    """

    def __init__(
        self,
        m2: float | nnx.Variable,
        space_shape: tuple[int, ...],
        channel_dim: int = 0,
        finite_size: bool = True,
        precompute_spectrum: bool = True,
        half: bool = True,
    ):
        self.half = half
        ks = fft_momenta(space_shape, lattice=finite_size)
        self.m2 = m2 if isinstance(m2, nnx.Variable) else Const(m2)
        if precompute_spectrum and not isinstance(m2, nnx.Variable):
            scaling = self.spectrum_function(ks, m2)
        else:
            scaling = None

        super().__init__(scaling, channel_dim=channel_dim)

    def spectrum_function(self, ks, m2):
        return jnp.sqrt((1 if self.half else 0.5) / (m2 + jnp.sum(ks**2, axis=-1)))

    @property
    def scaling(self):
        if self.scaling_var.value is None:
            return self.spectrum_function(self.ks, self.m2.value)
        return self.scaling_var.value


def get_fourier_masks(real_shape):
    """Get masks for independent d.o.f. of real FFT transform."""
    # rfft reduces last dimension
    rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

    real_mask = np.ones(rfft_shape, dtype=bool)
    imag_mask = np.ones(rfft_shape, dtype=bool)

    # right edge only if dimension is even length
    edges = [[0] + ([s // 2] if s % 2 == 0 else []) for s in real_shape]

    cp_start = [s // 2 + 1 for s in real_shape[:-1]]

    # zero because reality condition makes value real
    for index in product(*edges):
        imag_mask[index] = False

    # zero because degrees of freedom are duplicated
    for i, s in enumerate(cp_start):
        for e1 in product(*edges[:i]):
            for e2 in product(*edges[i + 1 :]):
                real_mask[e1 + (np.s_[s:],) + e2] = False
                imag_mask[e1 + (np.s_[s:],) + e2] = False

    return real_mask, imag_mask


def get_fourier_duplicated(real_shape):
    """Get indices of copied degrees of freedom in real FFT transform."""
    rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

    edges = [[0] + ([s // 2] if s % 2 == 0 else []) for s in real_shape]
    cp_start = [s // 2 + 1 for s in real_shape[:-1]]

    cp_from = []
    cp_to = []

    # degrees of freedom are duplicated
    for i, s in enumerate(cp_start):
        for e1 in product(*edges[:i]):
            for e2 in product(*edges[i + 1 :]):
                cp_from.extend(
                    [e1 + (-j % rfft_shape[i],) + e2 for j in range(s, rfft_shape[i])]
                )
                cp_to.extend([e1 + (j,) + e2 for j in range(s, rfft_shape[i])])

    return np.array(cp_from), np.array(cp_to)


def rfft_fold(rfft_values, real_shape):
    # get independent d.o.f.
    mr, mi = get_fourier_masks(real_shape)
    vr = rfft_values.real[..., mr]
    vi = rfft_values.imag[..., mi]
    return jnp.concatenate([vr, vi], axis=-1)


def rfft_unfold(values, real_shape):
    # get full rfft output matrix from independent d.o.f.
    mr, mi = get_fourier_masks(real_shape)
    copy_from, copy_to = get_fourier_duplicated(real_shape)

    vr, vi = jnp.split(values, [mr.sum()], axis=-1)
    vi = 1j * vi
    x = jnp.zeros(vi.shape[:-1] + mr.shape, dtype=vi.dtype)
    x = x.at[..., mr].set(vr)
    x = x.at[..., mi].add(vi)
    x = x.at[(np.s_[...], *copy_to.T)].set(x[(np.s_[...], *copy_from.T)].conj())
    return x
