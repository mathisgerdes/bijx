"""
Fourier utilities tests.

Covers: `fft_momenta`, `FourierMeta`, `FFTRep`, `FourierData`.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given

from bijx.fourier import FFTRep, FourierData, FourierMeta, fft_momenta

from .utils import (
    ATOL_RELAXED,
    RTOL_RELAXED,
    is_valid_array,
    random_real_arrays,
    real_space_shapes,
)


class TestFftMomenta:
    @pytest.mark.parametrize("shape", [(4,), (3, 5), (4, 4, 3)])
    @pytest.mark.parametrize("reduced", [True, False])
    @pytest.mark.parametrize("lattice", [True, False])
    def test_shapes_and_options(self, shape, reduced, lattice):
        k = fft_momenta(shape, reduced=reduced, lattice=lattice, unit=False)
        if reduced:
            exp_shape = shape[:-1] + (shape[-1] // 2 + 1, len(shape))
        else:
            exp_shape = shape + (len(shape),)
        assert k.shape == exp_shape
        assert is_valid_array(k)

    @pytest.mark.parametrize("shape", [(4,), (5,), (4, 4), (6, 5)])
    def test_momenta_folded_to_brillouin_zone(self, shape):
        """Momenta run over (-pi, pi], with mode n and L-n related by k -> -k.

        Folding is what makes ``|k|^2`` the momentum of a mode (mode ``L-1`` is
        the slowest non-zero mode, not the fastest) and what makes any spectrum
        built from these momenta satisfy ``SpectrumScaling``'s conjugate-pair
        precondition. Unfolded momenta silently break both.
        """
        k = np.asarray(fft_momenta(shape, reduced=False))
        assert (np.abs(k) <= np.pi + 1e-12).all()
        for ax, n in enumerate(shape):
            along = np.moveaxis(k[..., ax], ax, 0)
            along = along.reshape(n, -1)[:, 0]
            expected = (
                2
                * np.pi
                * np.where(2 * np.arange(n) > n, np.arange(n) - n, np.arange(n))
                / n
            )
            np.testing.assert_allclose(along, expected, atol=ATOL_RELAXED)
            # n and L-n carry opposite momentum, up to the identification of
            # the Nyquist mode's +pi and -pi (they are the same momentum)
            partner = along[(-np.arange(n)) % n]
            wrapped = np.mod(along + partner + np.pi, 2 * np.pi) - np.pi
            np.testing.assert_allclose(wrapped, 0.0, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("shape", [(8,), (4, 4), (8, 8), (6, 6, 6)])
    def test_ks_full_classes_respect_conjugate_pairs(self, shape):
        """Conjugate partners share a |k|^2 class, so a per-class spectrum is valid.

        ``ks_full`` folds its indices for exactly this reason: a spectrum
        parametrised through ``unique_unfold`` must give ``copy_from`` and
        ``copy_to`` the same value, or ``SpectrumScaling`` stops being a
        bijection once those parameters differ.
        """
        meta = FourierMeta.create(shape)
        ks = np.asarray(meta.ks_full)
        cf, ct = np.asarray(meta.copy_from), np.asarray(meta.copy_to)
        if len(ct) == 0:
            pytest.skip("no conjugate pairs stored in one space dimension")
        np.testing.assert_array_equal(ks[tuple(cf.T)], ks[tuple(ct.T)])

    def test_unit_indices(self):
        shape = (4, 6)
        k_idx = fft_momenta(shape, reduced=True, unit=True)
        exp_shape = shape[:-1] + (shape[-1] // 2 + 1, len(shape))
        assert k_idx.shape == exp_shape
        # Indices are integers in [0, n_i)
        assert (k_idx >= 0).all()
        for ax, n in enumerate(shape):
            assert (k_idx[..., ax] < n).all()


class TestFourierMeta:
    @given(real_space_shapes())
    def test_dof_conservation(self, real_shape):
        meta = FourierMeta.create(real_shape)
        total_dof = int(meta.mr.sum() + meta.mi.sum())
        assert total_dof == int(np.prod(real_shape))

    @given(real_space_shapes())
    def test_mask_shapes_and_types(self, real_shape):
        meta = FourierMeta.create(real_shape)
        exp = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        assert meta.mr.shape == exp
        assert meta.mi.shape == exp
        assert meta.mr.dtype == bool
        assert meta.mi.dtype == bool

    @given(real_space_shapes())
    def test_edge_imag_zero(self, real_shape):
        meta = FourierMeta.create(real_shape)
        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        edges = []
        for n in real_shape:
            e = [0]
            if n % 2 == 0:
                e.append(n // 2)
            edges.append(e)
        for idx in product(*edges):
            if idx[-1] < rfft_shape[-1]:
                assert not meta.mi[idx]


class TestFourierData:
    @pytest.mark.parametrize("rep", list(FFTRep))
    @given(random_real_arrays())
    def test_round_trip_real(self, rep, x_real):
        if not is_valid_array(x_real):
            return
        fd = FourierData.from_real(x_real, x_real.shape)
        fd_conv = fd.to(rep)
        fd_back = fd_conv.to(FFTRep.real_space)
        assert is_valid_array(fd_back.data)
        np.testing.assert_allclose(
            x_real, fd_back.data, atol=ATOL_RELAXED, rtol=RTOL_RELAXED
        )

    @given(random_real_arrays())
    def test_comp_real_size(self, x_real):
        fd = FourierData.from_real(x_real, x_real.shape, to=FFTRep.comp_real)
        assert fd.data.size == int(np.prod(x_real.shape))
