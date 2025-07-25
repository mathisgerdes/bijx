"""
Fourier-space bijections.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..fourier import FFTRep, FourierData, FourierMeta, fft_momenta, get_fourier_masks
from ..utils import Const, ShapeInfo
from .base import Bijection


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


class ToFourierData(Bijection):
    def __init__(self, real_shape, rep=None, channel_dim=0, unpack=False):
        self.meta = FourierMeta.create(real_shape, channel_dim)
        self.rep = rep
        self.unpack = unpack

    def forward(self, x, log_density, **kwargs):
        fft_data = FourierData(x, FFTRep.real_space, self.meta).to(self.rep)
        if self.unpack:
            fft_data = fft_data.data
        return fft_data, log_density

    def reverse(self, x, log_density, **kwargs):
        if self.unpack:
            x = FourierData(x, self.rep, self.meta)
        return x.to(FFTRep.real_space).data, log_density
