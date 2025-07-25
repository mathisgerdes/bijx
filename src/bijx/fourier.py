from enum import IntEnum
from itertools import product

import flax
import jax
import jax.numpy as jnp
import numpy as np

from .utils import ShapeInfo


def fft_momenta(
    shape: tuple[int, ...],
    reduced: bool = True,
    lattice: bool = False,
    unit: bool = False,
) -> jax.Array:
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


def get_fourier_masks(real_shape):
    """Get masks for independent d.o.f. of real FFT transform."""
    # rfft reduces last dimension
    rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

    # Start with all degrees of freedom available
    real_mask = np.ones(rfft_shape, dtype=bool)
    imag_mask = np.ones(rfft_shape, dtype=bool)

    # For a real-valued function, FFT satisfies F(k) = F*(-k)
    # This creates constraints on which coefficients are independent

    # 1. Reality constraints: at certain frequencies, coefficients must be real
    # This happens when k = -k (mod N), i.e., at 0 and N/2 (if N is even)

    # Frequencies where reality constraint applies
    edges = []
    for i, n in enumerate(real_shape):
        edge_freqs = [0]
        if n % 2 == 0:
            edge_freqs.append(n // 2)
        edges.append(edge_freqs)

    # At edge frequencies, imaginary part must be zero
    for edge_idx in product(*edges):
        # Check if this index is within the rFFT bounds
        if edge_idx[-1] < rfft_shape[-1]:
            imag_mask[edge_idx] = False

    # 2. Duplication constraints: some coefficients are conjugates of others
    # For rFFT, we need to identify which coefficients in the stored array
    # are related by Hermitian symmetry

    # Generate all frequency indices in the rFFT output
    for idx in np.ndindex(rfft_shape):
        k = np.array(idx)

        # Compute the conjugate frequency: -k mod N
        k_conj = np.array([(-ki) % ni for ki, ni in zip(k, real_shape)])

        # Check if the conjugate is also in the rFFT range
        if k_conj[-1] < rfft_shape[-1]:
            # Both k and k_conj are stored in rFFT
            # We need to keep only one and mark the other as dependent

            # Use lexicographic ordering to decide which one to keep
            k_tuple = tuple(k)
            k_conj_tuple = tuple(k_conj)

            if k_tuple != k_conj_tuple and k_tuple > k_conj_tuple:
                # This coefficient is a duplicate, mark as unavailable
                real_mask[idx] = False
                imag_mask[idx] = False

    return real_mask, imag_mask


def get_fourier_duplicated(real_shape):
    """Get indices of copied degrees of freedom in real FFT transform."""
    rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

    cp_from = []
    cp_to = []

    # Find all Hermitian symmetry relationships F(k) = F*(-k)
    for idx in np.ndindex(rfft_shape):
        k = np.array(idx)

        # Compute the conjugate frequency: -k mod N
        k_conj = np.array([(-ki) % ni for ki, ni in zip(k, real_shape)])

        # Check if the conjugate is also in the rFFT range
        if k_conj[-1] < rfft_shape[-1]:
            k_tuple = tuple(k)
            k_conj_tuple = tuple(k_conj)

            # If different indices are related by Hermitian symmetry,
            # record the relationship (choose consistent ordering)
            if k_tuple != k_conj_tuple and k_tuple > k_conj_tuple:
                cp_from.append(k_conj)
                cp_to.append(k)

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

    if len(copy_to) > 0:
        x = x.at[(np.s_[...], *copy_to.T)].set(x[(np.s_[...], *copy_from.T)].conj())

    return x


@flax.struct.dataclass
class FourierMeta:
    shape_info: ShapeInfo
    mr: np.ndarray = flax.struct.field(pytree_node=False)
    mi: np.ndarray = flax.struct.field(pytree_node=False)
    copy_from: np.ndarray = flax.struct.field(pytree_node=False)
    copy_to: np.ndarray = flax.struct.field(pytree_node=False)
    ks_full: np.ndarray = flax.struct.field(pytree_node=False)
    ks_reduced: np.ndarray = flax.struct.field(pytree_node=False)
    unique_idc: np.ndarray = flax.struct.field(
        pytree_node=False
    )  # unique values of |k|
    unique_unfold: np.ndarray = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, real_shape, channel_dim=0):
        mr, mi = get_fourier_masks(real_shape)
        copy_from, copy_to = get_fourier_duplicated(real_shape)

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
    real_space = 0  # 'real space data'
    rfft = 1  # 'raw rfft output'
    comp_complex = 2  # 'independent complex components'
    comp_real = 3  # 'independent real degrees of freedom'


@flax.struct.dataclass
class FourierData:
    data: jax.Array
    rep: FFTRep = flax.struct.field(pytree_node=False)
    meta: FourierMeta

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
