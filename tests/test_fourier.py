from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bijx.fourier import FFTRep, FourierData, FourierMeta

# Constants for numerical stability
ATOL_RECONSTRUCTION = 1e-6
RTOL_RECONSTRUCTION = 1e-5
ATOL_DOF = 1e-10

def is_valid_array(x: jnp.ndarray) -> bool:
    """Check if array contains valid values (no NaN or inf)."""
    return not (jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)))

@jax.jit
def count_total_dof(mr: jnp.ndarray, mi: jnp.ndarray) -> int:
    """JIT-compiled DOF counting."""
    return mr.sum() + mi.sum()

def generate_random_array(rng, shape: tuple[int, ...]) -> jnp.ndarray:
    """Generate random real-valued array using Hypothesis random."""
    data = np.array([rng.uniform(-1.0, 1.0) for _ in range(np.prod(shape))])
    return jnp.array(data.reshape(shape).astype(np.float32))

@st.composite
def real_shapes(draw) -> tuple[int, ...]:
    """Generate valid real-valued array shapes for FFT testing."""
    ndim = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(
        draw(st.integers(min_value=2, max_value=4)) for _ in range(ndim)
    )
    return shape

@st.composite
def real_arrays(draw) -> jnp.ndarray:
    """Generate random real-valued arrays for testing."""
    shape = draw(real_shapes())
    rng = draw(st.randoms())
    return generate_random_array(rng, shape)

class TestFourierMeta:
    """Tests for FourierMeta class."""

    @settings(deadline=None, max_examples=20)
    @given(real_shapes())
    def test_dof_conservation(self, real_shape: tuple[int, ...]) -> None:
        """Test that degrees of freedom are conserved."""
        meta = FourierMeta.create(real_shape)
        total_dof = count_total_dof(meta.mr, meta.mi)
        original_size = np.prod(real_shape)
        assert total_dof == original_size, f"DOF not conserved for shape {real_shape}: {total_dof} != {original_size}"

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_mask_shapes(self, real_shape: tuple[int, ...]) -> None:
        """Test that masks have correct shapes."""
        meta = FourierMeta.create(real_shape)
        expected_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        assert meta.mr.shape == expected_shape, f"Real mask shape incorrect: {meta.mr.shape} != {expected_shape}"
        assert meta.mi.shape == expected_shape, f"Imaginary mask shape incorrect: {meta.mi.shape} != {expected_shape}"
        assert meta.mr.dtype == bool, "Real mask should be boolean"
        assert meta.mi.dtype == bool, "Imaginary mask should be boolean"

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_edge_constraints(self, real_shape: tuple[int, ...]) -> None:
        """Test that edge frequencies have zero imaginary parts."""
        meta = FourierMeta.create(real_shape)
        edges = []
        for n in real_shape:
            edge_freqs = [0]
            if n % 2 == 0:
                edge_freqs.append(n // 2)
            edges.append(edge_freqs)
        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        for edge_idx in product(*edges):
            if edge_idx[-1] < rfft_shape[-1]:
                assert not meta.mi[edge_idx], f"Imaginary part should be False at edge {edge_idx}"

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_hermitian_relationships(self, real_shape: tuple[int, ...]) -> None:
        """Test that duplication relationships follow Hermitian symmetry."""
        meta = FourierMeta.create(real_shape)
        assert len(meta.copy_from) == len(meta.copy_to), "Mismatch in duplication arrays"
        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        for cp_from, cp_to in zip(meta.copy_from, meta.copy_to):
            assert all(0 <= idx < size for idx, size in zip(cp_from, rfft_shape)), f"copy_from index {cp_from} out of bounds for shape {rfft_shape}"
            assert all(0 <= idx < size for idx, size in zip(cp_to, rfft_shape)), f"copy_to index {cp_to} out of bounds for shape {rfft_shape}"
            k_conj = jnp.array([(-ki) % ni for ki, ni in zip(cp_to, real_shape)])
            np.testing.assert_array_equal(cp_from, k_conj, f"Not a valid Hermitian pair: {cp_from} -> {cp_to}")

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_consistency_with_masks(self, real_shape: tuple[int, ...]) -> None:
        """Test that duplicated indices are marked False in masks."""
        meta = FourierMeta.create(real_shape)
        for cp_idx in meta.copy_to:
            assert not meta.mr[tuple(cp_idx)], f"Duplicated index {cp_idx} should be False in real mask"
            assert not meta.mi[tuple(cp_idx)], f"Duplicated index {cp_idx} should be False in imaginary mask"

class TestFourierData:
    """Tests for FourierData class transformations."""

    @pytest.mark.parametrize("rep", list(FFTRep))
    @settings(deadline=None, max_examples=15)
    @given(real_arrays())
    def test_round_trip(self, rep: FFTRep, x_real: jnp.ndarray) -> None:
        """Test round-trip: real -> any representation -> real."""
        if not is_valid_array(x_real):
            return

        shape = x_real.shape
        fd_orig = FourierData.from_real(x_real, shape)

        fd_converted = fd_orig.to(rep)
        fd_reconstructed = fd_converted.to(FFTRep.real_space)

        if is_valid_array(fd_reconstructed.data):
            np.testing.assert_allclose(
                x_real,
                fd_reconstructed.data,
                atol=ATOL_RECONSTRUCTION,
                rtol=RTOL_RECONSTRUCTION,
                err_msg=f"Round-trip via {rep.name} failed",
            )

    @settings(deadline=None, max_examples=15)
    @given(real_arrays())
    def test_dof_sizes(self, x_real: jnp.ndarray) -> None:
        """Test that comp_real representation has correct DOF count."""
        if not is_valid_array(x_real):
            return
        shape = x_real.shape
        expected_dof = np.prod(shape)
        fd = FourierData.from_real(x_real, shape, to=FFTRep.comp_real)
        assert fd.data.size == expected_dof, f"comp_real size {fd.data.size} != expected DOF {expected_dof}"

# Specific regression tests for known cases
class TestRegressionCases:
    """Tests for specific cases that were previously broken."""

    def test_3d_dof_conservation(self) -> None:
        """Specific test for the (4,4,4) case that was failing."""
        shape = (4, 4, 4)
        meta = FourierMeta.create(shape)
        total_dof = count_total_dof(meta.mr, meta.mi)
        expected_dof = np.prod(shape)
        assert total_dof == expected_dof, f"DOF not conserved for {shape}: {total_dof} != {expected_dof}"

    def test_various_3d_cases(self) -> None:
        """Test various 3D cases that had issues."""
        test_shapes = [(3, 3, 3), (4, 4, 4), (4, 3, 2), (3, 3, 3)]
        for shape in test_shapes:
            meta = FourierMeta.create(shape)
            total_dof = count_total_dof(meta.mr, meta.mi)
            expected_dof = np.prod(shape)
            assert total_dof == expected_dof, f"DOF not conserved for {shape}: {total_dof} != {expected_dof}"

    def test_duplication_count_3d(self) -> None:
        """Test that we get the correct number of duplication pairs."""
        shape = (4, 4, 4)
        meta = FourierMeta.create(shape)
        assert len(meta.copy_from) == 12, f"Expected 12 duplication pairs for {shape}, got {len(meta.copy_from)}"
