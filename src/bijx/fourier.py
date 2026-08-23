r"""
Fourier transform utilities for lattice field theory and physics applications.

This module provides comprehensive utilities for working with Fourier transforms
of real-valued fields based on the FFT implementation in JAX.
"""

from dataclasses import replace
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
    "spectrum_asymmetry",
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
    r"""Generate momentum grid for Fourier transforms.

    Creates momentum coordinate arrays suitable for physics applications,
    supporting both continuum and lattice formulations. Handles the reduced
    form appropriate for real FFTs with Hermitian symmetry.

    Momenta are **folded into the first Brillouin zone**: mode $n$ of an axis of
    length $L$ carries momentum $2\pi n/L$ for $n \le L/2$ and $2\pi (n - L)/L$
    above that, so momenta run over $(-\pi, \pi]$ and modes $n$ and $L - n$ are
    related by $k \to -k$. Folding is what makes $|k|^2$ the momentum of that
    mode (mode $L-1$ is the *slowest* non-zero mode, not the fastest).
    With ``reduced=True`` the last axis only runs to $L/2$ and is unaffected;
    folding matters on every other axis, i.e. in two or more space dimensions.

    Args:
        shape: Spatial grid dimensions.
        reduced: If True, use reduced form for real FFT (Hermitian symmetry).
        lattice: If True, use lattice momenta; otherwise continuum momenta.
        unit: If True, return raw non-negative integer *indices* into the FFT
            grid instead of momenta. These are array indices, not momenta, and
            are deliberately **not** folded; for integer-valued momenta use
            ``fft_momenta(shape, unit=True)`` folded by the caller, or
            :attr:`FourierMeta.ks_full`, which does exactly that.

    Returns:
        Momentum grid array with shape ``(*spatial_shape, spatial_rank)``.
        For continuum: momenta in units of 2π/L, folded to $(-\pi, \pi]$.
        For lattice: $2\sin(k/2)$ of those momenta, odd in $k$.

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
    # fold into the first Brillouin zone: n > L/2 carries momentum 2 pi (n-L)/L.
    # Strict inequality keeps the Nyquist mode n = L/2 at +pi.
    ks = np.where(2 * ks > shape_factor, ks - shape_factor, ks)
    ks = 2 * jnp.pi * ks / shape_factor
    if lattice:
        # with this true, (finite) lattice spectrum ~ 1 / m^2 + k^2
        # otherwise get ~ 1 / k^2 - 2 (cos(2 pi k) - 1)
        ks = 2 * jnp.sin(ks / 2)
    # move "i" (space-dim index) to last axis
    return np.moveaxis(ks, 0, -1)


@nnx.dataclass
class FourierMeta(nnx.Pytree):
    r"""Metadata for handling real FFT constraints and symmetries.

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

    Note:
        ``ks_full`` is $|k|^2$ in units of $(2\pi/L)^2$, from indices folded into
        the first Brillouin zone, so conjugate partners (``copy_from``/
        ``copy_to``) always share a $|k|^2$ class. A spectrum parametrised per
        class through ``unique_idc``/``unique_unfold`` therefore satisfies the
        conjugate-pair precondition of :class:`~bijx.SpectrumScaling` by
        construction, learnable or not.
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
        # |k|^2 in units of (2 pi / L)^2, from indices folded into the first
        # Brillouin zone
        idx = np.asarray(fft_momenta(real_shape, unit=True))
        lengths = np.asarray(real_shape)
        ks_full = np.sum(np.minimum(idx, lengths - idx) ** 2, axis=-1).astype(int)
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


def spectrum_asymmetry(scaling, real_shape, channel_dim=0):
    r"""Tool: measure violation of the conjugate-pair precondition of a spectrum.

    A diagonal scaling in rFFT space defines an invertible map on real fields
    (with log-Jacobian $\sum_k w_k \log|s_k|$) only if two conditions hold:

    1. Entries whose conjugate partner is also stored in the rFFT grid are
       consistent, $s[\text{copy\_to}] == s[\text{copy\_from}]^*$.
    2. The spectrum is **real at self-conjugate modes** (``mr & ~mi``), where
       the field itself carries no imaginary degree of freedom.

    All other entries are unconstrained. See also :class:`~bijx.SpectrumScaling`.

    This is not checked inside the bijection (it cannot raise under ``jit``);
    call this function to assert the precondition in tests / debugging.

    Args:
        scaling: Spectrum with leading axes matching the rFFT shape of
            ``real_shape``, optionally followed by channel axes.
        real_shape: Shape of the real-space (spatial) data.
        channel_dim: Number of channel dimensions of ``scaling``.

    Returns:
        Maximum absolute violation of either condition; the larger of
        $\max |s[\text{copy\_to}] - s[\text{copy\_from}]^*|$ and
        $\max |\operatorname{Im} s|$ over the self-conjugate modes.

    Any spectrum built from :func:`fft_momenta` satisfies both conditions, since
    those momenta are folded and real; this function is for hand-built or
    externally supplied spectra.

    Example:
        >>> k = fft_momenta((6, 6))
        >>> spectrum = jnp.exp(-0.1 * jnp.sum(k**2, axis=-1))
        >>> assert spectrum_asymmetry(spectrum, (6, 6)) < 1e-12
        >>> asymmetric = spectrum.at[1, 0].multiply(1.8)  # breaks one pair
        >>> assert spectrum_asymmetry(asymmetric, (6, 6)) > 0
    """
    scaling = jnp.asarray(scaling)
    meta = FourierMeta.create(real_shape, channel_dim)
    space_dim = len(real_shape)

    if scaling.ndim not in (space_dim, space_dim + channel_dim):
        raise ValueError(
            f"scaling rank {scaling.ndim} matches neither space_dim={space_dim} "
            f"nor space_dim + channel_dim={space_dim + channel_dim}"
        )
    if scaling.shape[:space_dim] != meta.mr.shape:
        raise ValueError(
            f"leading axes of scaling {scaling.shape[:space_dim]} do not match "
            f"the rFFT shape {meta.mr.shape} of {tuple(real_shape)}"
        )

    abs_dtype = jnp.abs(jnp.zeros((), dtype=scaling.dtype)).dtype

    # condition 2. Non-trivial even in 1D
    self_conj = np.asarray(meta.mr) & ~np.asarray(meta.mi)
    if self_conj.any():
        imag = jnp.max(jnp.abs(jnp.imag(scaling[self_conj])))
    else:
        imag = jnp.zeros((), dtype=abs_dtype)

    # condition 1
    if len(meta.copy_to) == 0:
        return jnp.asarray(imag, dtype=abs_dtype)

    to = scaling[tuple(meta.copy_to.T)]
    frm = scaling[tuple(meta.copy_from.T)]
    pair = jnp.max(jnp.abs(to - jnp.conj(frm)))
    return jnp.maximum(
        jnp.asarray(pair, dtype=abs_dtype), jnp.asarray(imag, dtype=abs_dtype)
    )


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


@nnx.dataclass
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

    data: jax.Array = nnx.data()
    rep: FFTRep = nnx.static()
    meta: FourierMeta = nnx.data()

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
