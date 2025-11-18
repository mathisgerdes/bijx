r"""
Fourier transform utilities for lattice field theory and physics applications.

This module provides comprehensive utilities for working with Fourier transforms
of real-valued fields based on the FFT implementation in JAX.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils import ShapeInfo

__all__ = [
    # Core Fourier utilities
    "fft_momenta",
    "FourierMeta",
    "FFTRep",
    "FourierData",
]


def fft_momenta(
    shape: tuple[int, ...],
    reduced: bool = True,
    lattice: bool = False,
    unit: bool = False,
) -> jax.Array:
    """Generate momentum grid for Fourier transforms.

    Creates momentum coordinate arrays suitable for physics applications,
    supporting both continuum and lattice formulations. Handles the reduced
    form appropriate for real FFTs with Hermitian symmetry.

    Args:
        shape: Spatial grid dimensions.
        reduced: If True, use reduced form for real FFT (Hermitian symmetry).
        lattice: If True, use lattice momenta; otherwise continuum momenta.
        unit: If True, return integer indices instead of momentum values.

    Returns:
        Momentum grid array with shape ``(*spatial_shape, spatial_rank)``.
        For continuum: momenta in units of 2π/L.
        For lattice: momenta appropriate for lattice derivatives.

    Example:
        >>> # Continuum momenta for 2D lattice
        >>> k = fft_momenta((64, 64), lattice=False)
        >>> k_squared = jnp.sum(k**2, axis=-1)  # |k|²
        >>> # Lattice momenta for finite difference operators
        >>> k_lat = fft_momenta((64, 64), lattice=True)
    """
    shape_factor = np.reshape(shape, [-1] + [1] * len(shape))
    if reduced:
        # using reality condition, can eliminate about half of components
        shape = list(shape)[:-1] + [np.floor(shape[-1] / 2) + 1]

    # get frequencies divided by shape as large grid
    # ks[i] is k varying along axis i from 0 to L_i
    ks = np.mgrid[tuple(np.s_[:s] for s in shape)]
    if unit:
        return np.moveaxis(ks, 0, -1)
    ks = 2 * jnp.pi * ks / shape_factor
    if lattice:
        # with this true, (finite) lattice spectrum ~ 1 / m^2 + k^2
        # otherwise get ~ 1 / k^2 - 2 (cos(2 pi k) - 1)
        ks = 2 * jnp.sin(ks / 2)
    # move "i" (space-dim index) to last axis
    return np.moveaxis(ks, 0, -1)


@dataclass(frozen=True)
class FourierMeta(nnx.Pytree):
    """Metadata for handling real FFT constraints and symmetries.

    Encapsulates all the bookkeeping needed to work with real-valued Fourier
    transforms, including Hermitian symmetry constraints, multiplicities for
    log-Jacobian computation, and indexing for different representations.

    The metadata handles the reduction from full complex FFT to the independent
    real degrees of freedom.

    Args:
        shape_info: Shape information for spatial and channel dimensions.
        mr: Boolean mask for real (independent) Fourier modes.
        mi: Boolean mask for imaginary (independent) Fourier modes.
        copy_from: Indices of modes that are copied due to Hermitian symmetry.
        copy_to: Target indices for Hermitian symmetry copying.
        ks_full: Full momentum magnitude squared values.
        ks_reduced: Reduced momentum magnitude squared values.
        unique_idc: Indices of unique momentum magnitudes.
        unique_unfold: Mapping from reduced to unique momentum magnitudes.

    Note:
        This class is created automatically by :func:`FourierMeta.create()` and
        should usually not be instantiated directly.
    """

    shape_info: ShapeInfo
    mr: jax.Array
    mi: jax.Array
    copy_from: jax.Array
    copy_to: jax.Array
    ks_full: jax.Array
    ks_reduced: jax.Array
    unique_idc: jax.Array  # unique values of |k|
    unique_unfold: jax.Array

    def replace(self, **changes):
        """Create new config with specified parameters replaced."""
        return replace(self, **changes)

    @staticmethod
    def _get_fourier_info(real_shape):
        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

        real_mask = np.ones(rfft_shape, dtype=bool)
        imag_mask = np.ones(rfft_shape, dtype=bool)

        cp_from, cp_to = [], []

        # Enforce reality constraints for k = -k mod N (F(k) must be real)
        edges = [[0] if n % 2 != 0 else [0, n // 2] for n in real_shape]
        for edge_idx in product(*edges):
            if edge_idx[-1] < rfft_shape[-1]:
                imag_mask[edge_idx] = False

        # Enforce Hermitian symmetry F(k) = F*(-k) for other k
        for idx in np.ndindex(rfft_shape):
            k = np.array(idx)
            k_conj = np.array([(-ki) % ni for ki, ni in zip(k, real_shape)])

            # Check if conjugate is also within rFFT bounds
            if k_conj[-1] < rfft_shape[-1]:
                k_tuple, k_conj_tuple = tuple(k), tuple(k_conj)
                if k_tuple > k_conj_tuple:
                    real_mask[idx] = False
                    imag_mask[idx] = False
                    cp_from.append(k_conj)
                    cp_to.append(k)

        return real_mask, imag_mask, np.array(cp_from), np.array(cp_to)

    @classmethod
    def create(cls, real_shape, channel_dim=0):
        """Create FourierMeta for given real-space shape.

        Args:
            real_shape: Shape of real-space data.
            channel_dim: Number of channel dimensions.

        Returns:
            FourierMeta instance with all symmetry constraints computed.
        """
        mr, mi, copy_from, copy_to = cls._get_fourier_info(real_shape)
        ks_full = np.sum(fft_momenta(real_shape, unit=True) ** 2, axis=-1).astype(int)
        ks_reduced = ks_full[mr]

        # unique_idc -> assign to "k index" (could be used to add correlations)
        _, unique_idc, unique_unfold = np.unique(
            ks_reduced, return_index=True, return_inverse=True
        )

        return cls(
            shape_info=ShapeInfo(event_shape=real_shape, channel_dim=channel_dim),
            mr=mr,
            mi=mi,
            copy_from=copy_from,
            copy_to=copy_to,
            ks_full=ks_full,
            ks_reduced=ks_reduced,
            unique_idc=unique_idc,
            unique_unfold=unique_unfold,
        )

    @property
    def real_shape(self):
        return self.shape_info.space_shape

    @property
    def have_imag(self):
        return self.mi[self.mr]

    @property
    def channel_slices(self):
        return [np.s_[:]] * self.shape_info.channel_dim

    @property
    def idc_rfft_independent(self):
        return (np.s_[...], self.mr, *self.channel_slices)

    @property
    def idc_have_imag(self):
        return (np.s_[...], self.have_imag, *self.channel_slices)

    @property
    def idc_copy_from(self):
        return (np.s_[...], *self.copy_from.T, *self.channel_slices)

    @property
    def idc_copy_to(self):
        return (np.s_[...], *self.copy_to.T, *self.channel_slices)

    def get_complex_dtype(self, real_data):
        dtype = real_data.dtype
        out = jax.eval_shape(jnp.fft.rfft, jax.ShapeDtypeStruct((10,), dtype))
        return out.dtype


class FFTRep(IntEnum):
    """Enumeration of different Fourier data representations.

    Defines the various ways to represent Fourier data for real-valued fields,
    each with different trade-offs in terms of memory usage, computational
    efficiency, and mathematical convenience.

    Values:
        real_space: Original real-space field data.
        rfft: Raw output from real FFT (includes redundant information).
        comp_complex: Independent complex Fourier components only.
        comp_real: All independent real degrees of freedom as a single array.

    Note:
        The comp_real representation packs both real and imaginary parts
        of independent modes into a single real-valued array, maximizing
        compatibility with standard bijection layers.
    """

    real_space = 0  # 'real space data'
    rfft = 1  # 'raw rfft output'
    comp_complex = 2  # 'independent complex components'
    comp_real = 3  # 'independent real degrees of freedom'


@dataclass(frozen=True)
class FourierData(nnx.Pytree):
    """Multi-representation container for Fourier data.

    Provides a unified interface for working with Fourier data in different
    representations, with automatic conversion between formats. This enables
    seamless switching between representations based on computational needs.

    The container maintains the data, its current representation type, and
    the associated metadata needed for conversions. All conversions preserve
    the underlying mathematical content while changing the format.

    Args:
        data: The actual data array in the current representation.
        rep: Current representation type (FFTRep enum).
        meta: FourierMeta containing symmetry and indexing information.

    Example:
        >>> # Create from real-space data
        >>> fd = FourierData.from_real(x, (64, 64))
        >>> # Convert to complex components
        >>> fd_complex = fd.to(FFTRep.comp_complex)
        >>> # Convert to real degrees of freedom
        >>> fd_real = fd.to(FFTRep.comp_real)
    """

    data: nnx.Data[jax.Array]
    rep: FFTRep
    meta: FourierMeta

    def replace(self, **changes):
        """Create new config with specified parameters replaced."""
        return replace(self, **changes)

    @classmethod
    def from_real(cls, x, real_shape, to: FFTRep | None = None, channel_dim=0):
        meta = FourierMeta.create(real_shape, channel_dim)
        rep = FFTRep.real_space
        self = cls(x, rep, meta)
        if to is not None:
            self = self.to(to)
        return self

    def to(self, rep: FFTRep | None):

        if rep == self.rep or rep is None:
            return self

        if rep == FFTRep.real_space:
            self = self.to(FFTRep.rfft)
            return self.replace(
                data=self.rfft_to_real(self.data, self.meta),
                rep=FFTRep.real_space,
            )

        if rep == FFTRep.rfft:
            if self.rep == FFTRep.real_space:
                return self.replace(
                    data=self.real_to_rfft(self.data, self.meta),
                    rep=FFTRep.rfft,
                )
            else:
                self = self.to(FFTRep.comp_complex)
                return self.replace(
                    data=self.complex_to_rfft(self.data, self.meta),
                    rep=FFTRep.rfft,
                )

        if rep == FFTRep.comp_complex:
            if self.rep in {FFTRep.real_space, FFTRep.rfft}:
                self = self.to(FFTRep.rfft)
                return self.replace(
                    data=self.rfft_to_complex(self.data, self.meta),
                    rep=FFTRep.comp_complex,
                )
            else:
                self = self.to(FFTRep.comp_real)
                return self.replace(
                    data=self.rdof_to_complex(self.data, self.meta),
                    rep=FFTRep.comp_complex,
                )

        if rep == FFTRep.comp_real:
            self = self.to(FFTRep.comp_complex)
            return self.replace(
                data=self.complex_to_rdof(self.data, self.meta),
                rep=FFTRep.comp_real,
            )

        raise ValueError(f"Error converting from {self.rep} to {rep}")

    @staticmethod
    def rfft_to_real(rfft, meta):
        x = jnp.fft.irfftn(
            rfft, meta.real_shape, meta.shape_info.space_axes, norm="ortho"
        )
        return x

    @staticmethod
    def real_to_rfft(x, meta):
        rfft = jnp.fft.rfftn(
            x, meta.real_shape, meta.shape_info.space_axes, norm="ortho"
        )
        return rfft

    @staticmethod
    def complex_to_rfft(xk, meta):
        batch_shape = xk.shape[: -1 - meta.shape_info.channel_dim]
        if meta.shape_info.channel_dim == 0:
            channel_shape = ()
        else:
            channel_shape = xk.shape[-meta.shape_info.channel_dim :]

        rfft = jnp.zeros(batch_shape + meta.mr.shape + channel_shape, dtype=xk.dtype)
        rfft = rfft.at[..., meta.mr].set(xk)

        if len(meta.copy_to) > 0:
            rfft = rfft.at[meta.idc_copy_to].set(rfft[meta.idc_copy_from].conj())

        return rfft

    @staticmethod
    def rfft_to_complex(rfft, meta):
        comp = rfft[meta.idc_rfft_independent]
        return comp

    @staticmethod
    def rdof_to_complex(rdof, meta):
        real, imag = jnp.split(
            rdof, [meta.mr.sum()], axis=-1 - meta.shape_info.channel_dim
        )
        real = real.astype(meta.get_complex_dtype(real))
        xk = real.at[meta.idc_have_imag].add(1j * imag)
        return xk

    @staticmethod
    def complex_to_rdof(xk, meta):
        real = xk.real
        imag = xk.imag[meta.idc_have_imag]
        return jnp.concatenate([real, imag], axis=-1 - meta.shape_info.channel_dim)
