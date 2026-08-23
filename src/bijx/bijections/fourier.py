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
from .affine_complex import complex_affine_apply
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
        space_dim: Number of spatial dimensions. If None, inferred from the
            rank of ``scaling``, which assumes a single spectrum shared by
            all channels. Must be given explicitly for a per-channel
            spectrum (and whenever ``scaling`` is None).

    Note:
        The spatial part of the scaling array must have the same shape as the
        output of ``jnp.fft.rfftn`` over the space axes. With ``channel_dim > 0``
        the scaling may either have rank ``space_dim`` (one spectrum shared by
        all channels) or rank ``space_dim + channel_dim`` (per-channel spectra).

    Note:
        Those rFFT entries whose conjugate partner is also stored in the
        rFFT grid are constrained, and they must satisfy
        ``s[copy_to] == conj(s[copy_from])`` for the index pairs of
        :meth:`FourierMeta.create(space_shape) <FourierMeta.create>` (plain
        equality for a real spectrum). For a real spectrum the condition is vacuous in
        one dimension (no such pairs exist). It is separate from the other
        requirement that the spectrum is symmetric under $k \to -k$ everywhere.

        Violating it fails silently: ``mr + mi`` vanishes on every ``copy_to``
        entry, so the log-Jacobian weights assume the symmetry rather
        than check it. The map then stops being invertible *and* the reported
        log-density change is wrong. Use :func:`~bijx.fourier.spectrum_asymmetry`
        to check the precondition.

        Any real function of :func:`~bijx.fourier.fft_momenta` satisfies this
        automatically, since those momenta are folded into the first Brillouin zone
        and are therefore related by $k \to -k$ across each conjugate pair,
        as is a spectrum parametrised per $|k|^2$ class through
        ``FourierMeta.unique_unfold``.

        A complex spectrum carries a second condition: it must be real at
        self-conjugate modes (``mr & ~mi``), where the field carries no
        imaginary degree of freedom. See :func:`~bijx.fourier.spectrum_asymmetry`.

    Example:
        >>> # Create momentum-dependent scaling
        >>> k = fft_momenta((8, 8))
        >>> scaling = jnp.exp(-0.1 * jnp.sum(k**2, axis=-1))
        >>> bijection = SpectrumScaling(scaling)
        >>> y, log_det = bijection.forward(phi, log_density)
    """

    def __init__(
        self,
        scaling: jax.Array | nnx.Variable,
        channel_dim: int = 0,
        space_dim: int | None = None,
    ):
        self.channel_dim = channel_dim

        if not isinstance(scaling, nnx.Variable):
            scaling = Const(scaling)
        self.scaling_var = scaling
        if space_dim is None:
            value = scaling.get_value()
            if value is None:
                raise ValueError(
                    "space_dim must be given explicitly if scaling is None"
                )
            # bare spectrum: rank is the number of space dimensions
            space_dim = jnp.ndim(value)
        self.shape_info = ShapeInfo(space_dim=space_dim, channel_dim=channel_dim)

    @property
    def scaling(self):
        return self.scaling_var.get_value()

    def _broadcast_scaling(self, scaling, shape_info):
        """Align scaling with the rFFT event shape ``(*rfft_shape, *channels)``."""
        space_dim = shape_info.space_dim
        channel_dim = shape_info.channel_dim
        rank = jnp.ndim(scaling)
        if rank == space_dim:
            # one spectrum shared by all channels
            scaling = jnp.reshape(scaling, jnp.shape(scaling) + (1,) * channel_dim)
        elif rank != space_dim + channel_dim:
            raise ValueError(
                f"scaling rank {rank} matches neither space_dim={space_dim} "
                f"(shared spectrum) nor space_dim + channel_dim="
                f"{space_dim + channel_dim} (per-channel spectrum)"
            )
        return scaling

    def apply(self, x, log_density, reverse=False, **kwargs):
        _, shape_info = self.shape_info.process_event(x.shape)
        meta = FourierMeta.create(shape_info.space_shape)
        scaling = self._broadcast_scaling(self.scaling, shape_info)

        # full rFFT event shape: spectrum axes followed by channel axes
        event_shape = meta.mr.shape + shape_info.channel_shape
        # weight = number of independent real DoF per rFFT mode (0, 1 or 2)
        weight = (meta.mr.astype(int) + meta.mi.astype(int)).reshape(
            meta.mr.shape + (1,) * shape_info.channel_dim
        )
        # broadcast before summing: a shared spectrum acts on every channel,
        # so its contribution to log|det J| is counted once per channel
        log_scaling = jnp.broadcast_to(jnp.log(jnp.abs(scaling)), event_shape)
        delta_ld = jnp.sum(weight * log_scaling)

        x_k = jnp.fft.rfftn(x, shape_info.space_shape, shape_info.space_axes)
        x_k, log_density = complex_affine_apply(
            x_k,
            log_density,
            scale=scaling,
            delta_ld=delta_ld,
            invert=reverse,
        )
        x = jnp.fft.irfftn(x_k, shape_info.space_shape, shape_info.space_axes)
        return x, log_density


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
        self.ks = Const(fft_momenta(space_shape, lattice=finite_size))
        self.m2 = m2 if isinstance(m2, nnx.Variable) else Const(m2)
        if precompute_spectrum and not isinstance(m2, nnx.Variable):
            scaling = Const(self.spectrum_function(self.ks.get_value(), m2))
        else:
            scaling = None

        super().__init__(scaling, channel_dim=channel_dim, space_dim=len(space_shape))

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
        scaling_var = self.scaling_var.get_value()
        if scaling_var is None:
            return self.spectrum_function(self.ks.get_value(), self.m2.get_value())
        return scaling_var


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
