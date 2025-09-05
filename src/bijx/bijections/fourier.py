r"""
Fourier-space bijections for physics applications.

This module implements bijections that operate in Fourier space, particularly
useful when spatial symmetries are present (e.g. lattice field theory).

Key components:
    - ``SpectrumScaling``: Diagonal scaling in Fourier space
    - ``FreeTheoryScaling``: Physics-motivated spectral scaling
    - ``ToFourierData``: Conversion between real and Fourier representations
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..fourier import FFTRep, FourierData, FourierMeta, fft_momenta
from ..utils import Const, ShapeInfo
from .base import ApplyBijection, Bijection


class SpectrumScaling(ApplyBijection):
    r"""Diagonal scaling transformation in Fourier space.

    Applies element-wise scaling to the Fourier transform of real-valued fields,
    implementing diagonal transformations in momentum space. This is particularly
    useful for implementing free field theories and spectral preconditioning.

    Type: $\mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H \times W \times C}$
    Transform: $\mathcal{F}^{-1}[s(\mathbf{k}) \mathcal{F}[\mathbf{x}]]$

    The scaling factors correspond to momentum-dependent transformations, with
    the log-Jacobian computed from FFT multiplicities to handle real FFT symmetries.

    Args:
        scaling: Scaling factors with shape matching rFFT output.
            If not an nnx.Variable/nnx.Param, by default treated as constant.
        channel_dim: Number of channel dimensions.

    Note:
        The scaling array must have the same shape as the output of jnp.fft.rfftn
        to ensure proper broadcasting during the Fourier-space multiplication.

    Example:
        >>> # Create momentum-dependent scaling
        >>> k = fft_momenta((8, 8))
        >>> scaling = jnp.exp(-0.1 * jnp.sum(k**2, axis=-1))
        >>> bijection = SpectrumScaling(scaling)
        >>> y, log_det = bijection.forward(phi, log_density)
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
        """Apply Fourier-space scaling transformation.

        Transforms the input through FFT, applies scaling, and transforms back.
        Computes the log-Jacobian contribution from the scaling factors.

        Args:
            r: Input array to transform.
            reverse: If True, apply inverse scaling (division).

        Returns:
            Tuple of (transformed_array, log_jacobian_contribution).
        """
        _, shape_info = self.shape_info.process_event(r.shape)
        meta = FourierMeta.create(shape_info.space_shape)
        r = jnp.fft.rfftn(r, shape_info.space_shape, shape_info.space_axes)
        r = r / self.scaling if reverse else r * self.scaling
        r = jnp.fft.irfftn(r, shape_info.space_shape, shape_info.space_axes)

        factor = meta.mr.astype(int) + meta.mi.astype(int)
        delta_ld = jnp.sum(factor * jnp.log(jnp.abs(self.scaling)))

        return r, delta_ld

    def apply(self, x, log_density, reverse=False, **kwargs):
        x, delta = self.scale(x, reverse=reverse)
        return x, log_density - delta if reverse else log_density + delta


class FreeTheoryScaling(SpectrumScaling):
    r"""Scaling bijection mapping white noise to free field theory.

    Implements the momentum-space scaling transformation that converts Gaussian
    white noise into samples from a free scalar field theory.

    Type: $\mathbb{R}^{H \times W \times C} \to \mathbb{R}^{H \times W \times C}$

    Transform:

    $$
    \mathcal{F}^{-1}\left[\frac{1}{\sqrt{m^2 + \mathbf{k}^2}} \mathcal{F}[\xi]\right]
    $$

    For a free scalar field with mass $m$, the two-point correlation function in
    momentum space is

    $$
    \langle\tilde{\phi}(\mathbf{k})\tilde{\phi}^*(\mathbf{k}')\rangle =
    \frac{\delta(\mathbf{k}-\mathbf{k}')}{m^2 + \mathbf{k}^2}\,.
    $$

    Args:
        m2: Mass squared parameter (can be learnable nnx.Variable).
        space_shape: Spatial lattice dimensions.
        channel_dim: Number of channel dimensions.
        finite_size: Whether to use lattice momenta (True) or continuum (False).
        precompute_spectrum: Whether to precompute scaling factors.
        half: Whether to use factor of 1/2 conventional in the action.

    Note:
        Assumes periodic boundary conditions.

    Example:
        >>> # Free scalar field with mass m=0.1 on 32x32 lattice
        >>> m2 = 0.01
        >>> scaling = FreeTheoryScaling(
        ...     m2, space_shape=(32, 32), finite_size=True
        ... )
        >>> phi, log_det = scaling.forward(eps, log_density)  # eps ~ N(0,1)
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
            scaling = Const(self.spectrum_function(ks, m2))
        else:
            scaling = None

        super().__init__(scaling, channel_dim=channel_dim)

    def spectrum_function(self, ks, m2):
        """Compute free field scaling factors from momenta and mass.

        Args:
            ks: Momentum grid with shape (..., spatial_dim).
            m2: Mass squared parameter.

        Returns:
            Scaling factors for free field theory propagator.
        """
        return jnp.sqrt((1 if self.half else 0.5) / (m2 + jnp.sum(ks**2, axis=-1)))

    @property
    def scaling(self):
        """Multiplicative factor in Fourier space."""
        if self.scaling_var.value is None:
            return self.spectrum_function(self.ks, self.m2.value)
        return self.scaling_var.value


class ToFourierData(Bijection):
    r"""Bijection for converting between real and Fourier data representations.

    Provides a bijective transformation between real-space arrays and their
    various Fourier representations (FFT output, independent components, etc.).
    This is useful for working with different Fourier data formats within flows.

    Type: $\mathbb{R}^{\text{real_shape}} \leftrightarrow \text{FourierData}$
    Transform: Format conversion between real and Fourier representations

    Args:
        real_shape: Shape of the real-space data.
        rep: Target Fourier representation (FFTRep enum value).
        channel_dim: Number of channel dimensions.
        unpack: Whether to unpack FourierData to raw array.

    Note:
        The Fourier transform can be understood as a rotation, and is therefore
        volume-preserving. The log-density remains unchanged.

    Example:
        >>> # Convert to complex Fourier components
        >>> converter = ToFourierData((32, 32), rep=FFTRep.comp_complex)
        >>> fourier_data, log_det = converter.forward(real_data, log_density)
        >>> # log_det == 0 (volume-preserving)
    """

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
