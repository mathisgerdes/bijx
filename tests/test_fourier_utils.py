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
