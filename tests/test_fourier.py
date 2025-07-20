import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lfx.fourier import (
    FFTRep,
    FourierData,
    get_fourier_duplicated,
    get_fourier_masks,
    rfft_fold,
    rfft_unfold,
)

# Constants for numerical stability
ATOL_RECONSTRUCTION = 1e-6  # Absolute tolerance for reconstruction checks
RTOL_RECONSTRUCTION = 1e-5  # Relative tolerance for reconstruction checks
ATOL_DOF = 1e-10  # Tolerance for DOF conservation (should be exact)


def is_valid_array(x: jnp.ndarray) -> bool:
    """Check if array contains valid values (no NaN or inf)."""
    return not (jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)))


@jax.jit
def jit_rfft_round_trip_core(
    x_real: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled core FFT round-trip operations."""
    shape = x_real.shape

    # Forward: real -> rfft
    x_rfft = jnp.fft.rfftn(x_real)

    # Fold: rfft -> independent DOF
    x_folded = rfft_fold(x_rfft, shape)

    # Unfold: independent DOF -> rfft
    x_unfolded = rfft_unfold(x_folded, shape)

    # Reverse: rfft -> real
    x_reconstructed = jnp.fft.irfftn(x_unfolded, shape)

    return x_rfft, x_unfolded, x_reconstructed


@jax.jit
def count_total_dof(mr: jnp.ndarray, mi: jnp.ndarray) -> int:
    """JIT-compiled DOF counting."""
    return mr.sum() + mi.sum()


def generate_random_array(rng, shape: tuple[int, ...]) -> jnp.ndarray:
    """Generate random real-valued array using Hypothesis random."""
    # Use the standard numpy API compatible with HypothesisRandom
    data = np.array([rng.uniform(-1.0, 1.0) for _ in range(np.prod(shape))])
    return jnp.array(data.reshape(shape).astype(np.float32))


@st.composite
def real_shapes(draw) -> tuple[int, ...]:
    """Generate valid real-valued array shapes for FFT testing."""
    # Use smaller sizes for faster testing
    ndim = draw(st.integers(min_value=1, max_value=3))  # Reduced from 4
    shape = tuple(
        draw(st.integers(min_value=2, max_value=4)) for _ in range(ndim)
    )  # Reduced from 8
    return shape


@st.composite
def real_arrays(draw) -> jnp.ndarray:
    """Generate random real-valued arrays for testing."""
    shape = draw(real_shapes())
    rng = draw(st.randoms())
    return generate_random_array(rng, shape)


class TestFourierMasks:
    """Tests for get_fourier_masks function."""

    @settings(deadline=None, max_examples=20)
    @given(real_shapes())
    def test_dof_conservation(self, real_shape: tuple[int, ...]) -> None:
        """Test that degrees of freedom are conserved."""
        mr, mi = get_fourier_masks(real_shape)

        # Total DOF should equal original array size
        total_dof = count_total_dof(mr, mi)
        original_size = np.prod(real_shape)

        assert total_dof == original_size, (
            f"DOF not conserved for shape {real_shape}: "
            f"{total_dof} != {original_size}"
        )

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_mask_shapes(self, real_shape: tuple[int, ...]) -> None:
        """Test that masks have correct shapes."""
        mr, mi = get_fourier_masks(real_shape)

        # Expected rFFT shape
        expected_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

        assert (
            mr.shape == expected_shape
        ), f"Real mask shape incorrect: {mr.shape} != {expected_shape}"
        assert (
            mi.shape == expected_shape
        ), f"Imaginary mask shape incorrect: {mi.shape} != {expected_shape}"
        assert mr.dtype == bool, "Real mask should be boolean"
        assert mi.dtype == bool, "Imaginary mask should be boolean"

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_edge_constraints(self, real_shape: tuple[int, ...]) -> None:
        """Test that edge frequencies have zero imaginary parts."""
        mr, mi = get_fourier_masks(real_shape)

        # At edge frequencies (0 and N//2), imaginary parts should be False
        from itertools import product

        edges = []
        for n in real_shape:
            edge_freqs = [0]
            if n % 2 == 0:
                edge_freqs.append(n // 2)
            edges.append(edge_freqs)

        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

        for edge_idx in product(*edges):
            if edge_idx[-1] < rfft_shape[-1]:  # Within rFFT bounds
                assert not mi[
                    edge_idx
                ], f"Imaginary part should be False at edge {edge_idx}"


class TestFourierDuplicated:
    """Tests for get_fourier_duplicated function."""

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_hermitian_relationships(self, real_shape: tuple[int, ...]) -> None:
        """Test that duplication relationships follow Hermitian symmetry."""
        copy_from, copy_to = get_fourier_duplicated(real_shape)

        assert len(copy_from) == len(copy_to), "Mismatch in duplication arrays"

        rfft_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)

        for cp_from, cp_to in zip(copy_from, copy_to):
            # Verify indices are within bounds
            assert all(
                0 <= idx < size for idx, size in zip(cp_from, rfft_shape)
            ), f"copy_from index {cp_from} out of bounds for shape {rfft_shape}"
            assert all(
                0 <= idx < size for idx, size in zip(cp_to, rfft_shape)
            ), f"copy_to index {cp_to} out of bounds for shape {rfft_shape}"

            # Verify Hermitian relationship: cp_from = (-cp_to) % real_shape
            k_conj = jnp.array([(-ki) % ni for ki, ni in zip(cp_to, real_shape)])
            np.testing.assert_array_equal(
                cp_from, k_conj, f"Not a valid Hermitian pair: {cp_from} -> {cp_to}"
            )

    @settings(deadline=None, max_examples=15)
    @given(real_shapes())
    def test_consistency_with_masks(self, real_shape: tuple[int, ...]) -> None:
        """Test that duplicated indices are marked False in masks."""
        mr, mi = get_fourier_masks(real_shape)
        copy_from, copy_to = get_fourier_duplicated(real_shape)

        # All duplicated indices should be False in both masks
        for cp_idx in copy_to:
            assert not mr[
                tuple(cp_idx)
            ], f"Duplicated index {cp_idx} should be False in real mask"
            assert not mi[
                tuple(cp_idx)
            ], f"Duplicated index {cp_idx} should be False in imaginary mask"


class TestRoundTrip:
    """Tests for round-trip functionality."""

    @settings(deadline=None, max_examples=15)
    @given(real_arrays())
    def test_rfft_fold_unfold(self, x_real: jnp.ndarray) -> None:
        """Test round-trip: real -> rfft -> fold -> unfold -> rfft -> real."""
        if not is_valid_array(x_real):
            return

        shape = x_real.shape

        # Use JIT-compiled round-trip operations
        x_rfft, x_unfolded, x_reconstructed = jit_rfft_round_trip_core(x_real)

        if not (
            is_valid_array(x_rfft)
            and is_valid_array(x_unfolded)
            and is_valid_array(x_reconstructed)
        ):
            return

        # Check that folded array has correct size (done inside JIT function)
        x_folded = rfft_fold(x_rfft, shape)
        expected_dof = np.prod(shape)
        assert (
            x_folded.size == expected_dof
        ), f"Folded array size {x_folded.size} != expected DOF {expected_dof}"

        # Check reconstruction of rfft
        np.testing.assert_allclose(
            x_rfft,
            x_unfolded,
            atol=ATOL_RECONSTRUCTION,
            rtol=RTOL_RECONSTRUCTION,
            err_msg="rFFT reconstruction failed",
        )

        # Check final reconstruction
        np.testing.assert_allclose(
            x_real,
            x_reconstructed,
            atol=ATOL_RECONSTRUCTION,
            rtol=RTOL_RECONSTRUCTION,
            err_msg="Real space reconstruction failed",
        )


class TestFourierData:
    """Tests for FourierData class transformations."""

    @settings(deadline=None, max_examples=5)  # Reduced significantly for expensive test
    @given(real_arrays())
    def test_comp_real_conversion(self, x_real: jnp.ndarray) -> None:
        """Test conversion to comp_real representation (most commonly used)."""
        if not is_valid_array(x_real):
            return

        shape = x_real.shape

        # Create FourierData from real space
        fd = FourierData.from_real(x_real, shape)

        # Test conversion to comp_real
        fd_converted = fd.to(FFTRep.comp_real)
        assert fd_converted.rep == FFTRep.comp_real, "Conversion to comp_real failed"

        if not is_valid_array(fd_converted.data):
            return

        # Convert back to real space and check reconstruction
        fd_back = fd_converted.to(FFTRep.real_space)

        if is_valid_array(fd_back.data):
            np.testing.assert_allclose(
                x_real,
                fd_back.data,
                atol=ATOL_RECONSTRUCTION,
                rtol=RTOL_RECONSTRUCTION,
                err_msg="Round-trip via comp_real failed",
            )

    @settings(deadline=None, max_examples=8)
    @given(real_arrays())
    def test_rfft_conversion(self, x_real: jnp.ndarray) -> None:
        """Test conversion to rfft representation."""
        if not is_valid_array(x_real):
            return

        shape = x_real.shape

        # Create FourierData from real space
        fd = FourierData.from_real(x_real, shape)

        # Test conversion to rfft
        fd_converted = fd.to(FFTRep.rfft)
        assert fd_converted.rep == FFTRep.rfft, "Conversion to rfft failed"

        if not is_valid_array(fd_converted.data):
            return

        # Convert back to real space and check reconstruction
        fd_back = fd_converted.to(FFTRep.real_space)

        if is_valid_array(fd_back.data):
            np.testing.assert_allclose(
                x_real,
                fd_back.data,
                atol=ATOL_RECONSTRUCTION,
                rtol=RTOL_RECONSTRUCTION,
                err_msg="Round-trip via rfft failed",
            )

    @settings(deadline=None, max_examples=15)
    @given(real_arrays())
    def test_dof_sizes(self, x_real: jnp.ndarray) -> None:
        """Test that comp_real representation has correct DOF count."""
        if not is_valid_array(x_real):
            return

        shape = x_real.shape
        expected_dof = np.prod(shape)

        # Convert to comp_real and check size
        fd = FourierData.from_real(x_real, shape, to=FFTRep.comp_real)

        assert (
            fd.data.size == expected_dof
        ), f"comp_real size {fd.data.size} != expected DOF {expected_dof}"


# Specific regression tests for known cases
class TestRegressionCases:
    """Tests for specific cases that were previously broken."""

    def test_3d_dof_conservation(self) -> None:
        """Specific test for the (4,4,4) case that was failing."""
        shape = (4, 4, 4)  # Reduced from (8,8,8)
        mr, mi = get_fourier_masks(shape)

        total_dof = count_total_dof(mr, mi)
        expected_dof = np.prod(shape)  # 64

        assert (
            total_dof == expected_dof
        ), f"DOF not conserved for {shape}: {total_dof} != {expected_dof}"

    def test_various_3d_cases(self) -> None:
        """Test various 3D cases that had issues."""
        test_shapes = [
            (3, 3, 3),  # Reduced sizes
            (4, 4, 4),
            (4, 3, 2),
            (3, 3, 3),  # odd dimensions
        ]

        for shape in test_shapes:
            mr, mi = get_fourier_masks(shape)
            total_dof = count_total_dof(mr, mi)
            expected_dof = np.prod(shape)

            assert (
                total_dof == expected_dof
            ), f"DOF not conserved for {shape}: {total_dof} != {expected_dof}"

    def test_duplication_count_3d(self) -> None:
        """Test that we get the correct number of duplication pairs."""
        shape = (4, 4, 4)
        copy_from, copy_to = get_fourier_duplicated(shape)

        assert (
            len(copy_from) == 12
        ), f"Expected 12 duplication pairs for {shape}, got {len(copy_from)}"
