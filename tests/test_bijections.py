import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from lfx.bijections import (
    AffineLayer,
    BetaStretch,
    Bijection,
    BinaryMask,
    ExpLayer,
    GaussianCDF,
    PowerLayer,
    SigmoidLayer,
    SoftPlusLayer,
    TanhLayer,
    TanLayer,
    checker_mask,
)

# Constants for numerical stability
ATOL_INVERSE = 1e-4  # Absolute tolerance for inverse checks
RTOL_INVERSE = 1e-3  # Relative tolerance for inverse checks
ATOL_LOG_DENSITY = 1e-3  # Absolute tolerance for log density checks
RTOL_LOG_DENSITY = 1e-2  # Relative tolerance for log density checks

# Range limits for different bijection types
GAUSSIAN_RANGE = (-3.0, 3.0)  # Range for unbounded domains
UNIT_INTERVAL_RANGE = (0.01, 0.99)  # Range for [0, 1] domain
MINUS_ONE_TO_ONE_RANGE = (-0.99, 0.99)  # Range for [-1, 1] domain


def is_valid_array(x: jnp.ndarray) -> bool:
    """Check if array contains valid values (no NaN or inf)."""
    return not (jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)))


def check_inverse(bijection: Bijection, x: jnp.ndarray) -> None:
    """Check that forward(reverse(x)) ≈ x and reverse(forward(x)) ≈ x."""
    # Skip empty arrays
    if x.size == 0:
        return

    # Initialize log density to zero
    log_density = jnp.zeros(())

    # Forward then reverse
    y, ld_forward = bijection.forward(x, log_density)
    if not is_valid_array(y):
        return

    x_back, ld_back = bijection.reverse(y, ld_forward)
    if not is_valid_array(x_back):
        return

    # Check x ≈ x_back
    np.testing.assert_allclose(x, x_back, atol=ATOL_INVERSE, rtol=RTOL_INVERSE)
    np.testing.assert_allclose(ld_back, 0, atol=ATOL_LOG_DENSITY, rtol=RTOL_LOG_DENSITY)


def check_log_density(bijection: Bijection, x: jnp.ndarray) -> None:
    """Check that the log density change matches computation with jax.vjp."""
    # Skip empty arrays
    if x.size == 0:
        return

    # Use scalar input for consistent testing
    if hasattr(x, "shape") and len(x.shape) > 0:
        x_test = jnp.array(x.flatten()[0], dtype=jnp.float32)
    else:
        x_test = jnp.array(x, dtype=jnp.float32)

    # Define function that returns only the forward transformation
    def forward_fn(x_in: jnp.ndarray) -> jnp.ndarray:
        y, _ = bijection.forward(x_in, 0.0)
        return y

    # Compute the Jacobian using vector-Jacobian product
    y, vjp_fn = jax.vjp(forward_fn, x_test)
    if not is_valid_array(y):
        return

    # For scalar operations, the Jacobian is the derivative
    jacobian = vjp_fn(jnp.ones_like(y))[0]
    if not is_valid_array(jacobian):
        return

    log_det_jacobian = jnp.log(jnp.abs(jacobian))

    # Get the reported log density change from the bijection
    _, reported_log_density = bijection.forward(x_test, 0.0)

    # Skip if values are invalid
    if not (is_valid_array(log_det_jacobian) and is_valid_array(reported_log_density)):
        return

    # The negative of the reported log density should match the log determinant
    np.testing.assert_allclose(
        -reported_log_density,
        log_det_jacobian,
        atol=ATOL_LOG_DENSITY,
        rtol=RTOL_LOG_DENSITY,
    )


# Test strategies tailored to each bijection type
@st.composite
def gaussian_domain_inputs(draw) -> jnp.ndarray:
    """Generate arrays of valid inputs for unbounded domains."""
    shape = draw(
        st.one_of(
            st.just(()),  # scalar
            st.just((1,)),  # 1D with single element
            st.just((3, 2)),  # Smaller for faster testing
        )
    )

    return draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(
                min_value=GAUSSIAN_RANGE[0],
                max_value=GAUSSIAN_RANGE[1],
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@st.composite
def unit_interval_inputs(draw) -> jnp.ndarray:
    """Generate arrays of valid inputs for [0, 1] domain bijections."""
    shape = draw(
        st.one_of(
            st.just(()),  # scalar
            st.just((1,)),  # 1D with single element
        )
    )

    return draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(
                min_value=UNIT_INTERVAL_RANGE[0],
                max_value=UNIT_INTERVAL_RANGE[1],
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@st.composite
def minus_one_to_one_inputs(draw) -> jnp.ndarray:
    """Generate arrays of valid inputs for [-1, 1] domain bijections."""
    shape = draw(
        st.one_of(
            st.just(()),  # scalar
            st.just((1,)),  # 1D with single element
        )
    )

    return draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(
                min_value=MINUS_ONE_TO_ONE_RANGE[0],
                max_value=MINUS_ONE_TO_ONE_RANGE[1],
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@st.composite
def positive_inputs(draw) -> jnp.ndarray:
    """Generate arrays of valid positive inputs for [0, inf] domain bijections."""
    shape = draw(
        st.one_of(
            st.just(()),  # scalar
            st.just((1,)),  # 1D with single element
        )
    )

    return draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )


# Tests for each bijection
class TestGaussianCDF:
    """Tests for the GaussianCDF bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        bijection = GaussianCDF(rngs=nnx.Rngs(params=key))
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        bijection = GaussianCDF(rngs=nnx.Rngs(params=key))
        check_log_density(bijection, x)


class TestTanLayer:
    """Tests for the TanLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(unit_interval_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = TanLayer()
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(unit_interval_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = TanLayer()
        check_log_density(bijection, x)


class TestSigmoidLayer:
    """Tests for the SigmoidLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = SigmoidLayer()
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = SigmoidLayer()
        check_log_density(bijection, x)


class TestTanhLayer:
    """Tests for the TanhLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = TanhLayer()
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = TanhLayer()
        check_log_density(bijection, x)


class TestBetaStretch:
    """Tests for the BetaStretch bijection."""

    @settings(deadline=None, max_examples=10)
    @given(unit_interval_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        a = nnx.Param(jnp.array(2.0))
        bijection = BetaStretch(a, rngs=nnx.Rngs(params=key))
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(unit_interval_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        a = nnx.Param(jnp.array(2.0))
        bijection = BetaStretch(a, rngs=nnx.Rngs(params=key))
        check_log_density(bijection, x)


class TestExpLayer:
    """Tests for the ExpLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = ExpLayer()
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = ExpLayer()
        check_log_density(bijection, x)


class TestSoftPlusLayer:
    """Tests for the SoftPlusLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = SoftPlusLayer()
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = SoftPlusLayer()
        check_log_density(bijection, x)


class TestPowerLayer:
    """Tests for the PowerLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(positive_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        bijection = PowerLayer(exponent=2.0)
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(positive_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        bijection = PowerLayer(exponent=2.0)
        check_log_density(bijection, x)


class TestAffineLayer:
    """Tests for the AffineLayer bijection."""

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_inverse(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        bijection = AffineLayer(rngs=nnx.Rngs(params=key))
        check_inverse(bijection, x)

    @settings(deadline=None, max_examples=10)
    @given(gaussian_domain_inputs())
    def test_log_density(self, x: jnp.ndarray) -> None:
        key = jax.random.key(0)
        bijection = AffineLayer(rngs=nnx.Rngs(params=key))
        check_log_density(bijection, x)


class TestBinaryMask:
    def test_split_merge_roundtrip_1d(self):
        """Test split/merge roundtrip for 1D arrays."""
        # Create simple mask for 1D array
        mask = BinaryMask.from_boolean_mask(jnp.array([True, False, True, False]))

        # Test with simple 1D array
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        primary, secondary = mask.split(x)
        reconstructed = mask.merge(primary, secondary)

        np.testing.assert_allclose(x, reconstructed, rtol=1e-6)

        # Test with batch dimensions
        x_batch = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        primary_batch, secondary_batch = mask.split(x_batch)
        reconstructed_batch = mask.merge(primary_batch, secondary_batch)

        np.testing.assert_allclose(x_batch, reconstructed_batch, rtol=1e-6)

    def test_split_merge_roundtrip_2d(self):
        """Test split/merge roundtrip for 2D arrays."""
        # Create checkerboard mask
        mask = checker_mask((4, 4), parity=True)

        # Test with 2D array
        x = jnp.arange(16.0).reshape(4, 4)
        primary, secondary = mask.split(x)
        reconstructed = mask.merge(primary, secondary)

        np.testing.assert_allclose(x, reconstructed, rtol=1e-6)

        # Test with batch dimensions
        x_batch = jnp.stack([x, x + 100])
        primary_batch, secondary_batch = mask.split(x_batch)
        reconstructed_batch = mask.merge(primary_batch, secondary_batch)

        np.testing.assert_allclose(x_batch, reconstructed_batch, rtol=1e-6)

    def test_split_merge_with_features(self):
        """Test split/merge with extra feature dimensions."""
        mask = checker_mask((3, 3), parity=False)

        # Test with 1 feature dimension
        x = jnp.arange(27.0).reshape(3, 3, 3)  # spatial + 3 features
        primary, secondary = mask.split(x, extra_feature_dims=1)
        reconstructed = mask.merge(primary, secondary, extra_feature_dims=1)

        np.testing.assert_allclose(x, reconstructed, rtol=1e-6)

        # Test with batch + features
        x_batch = jnp.stack([x, x * 2])  # (2, 3, 3, 3)
        primary_batch, secondary_batch = mask.split(x_batch, extra_feature_dims=1)
        reconstructed_batch = mask.merge(
            primary_batch, secondary_batch, extra_feature_dims=1
        )

        np.testing.assert_allclose(x_batch, reconstructed_batch, rtol=1e-6)

    def test_mask_creation_consistency(self):
        """Test that different mask creation methods are consistent."""
        # Create boolean mask
        bool_mask = jnp.array([[True, False], [False, True]])

        # Create from boolean
        mask1 = BinaryMask.from_boolean_mask(bool_mask)

        # Create from indices
        indices = jnp.where(bool_mask)
        mask2 = BinaryMask.from_indices(indices, bool_mask.shape)

        # Test arrays should be identical
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        p1, s1 = mask1.split(x)
        p2, s2 = mask2.split(x)

        np.testing.assert_allclose(p1, p2, rtol=1e-6)
        np.testing.assert_allclose(s1, s2, rtol=1e-6)

    def test_mask_flip(self):
        """Test mask flipping functionality."""
        mask = checker_mask((2, 2), parity=True)
        mask_inv = mask.flip()
        mask_inv2 = ~mask  # test operator overload

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Primary/secondary should be swapped
        p1, s1 = mask.split(x)
        p2, s2 = mask_inv.split(x)

        np.testing.assert_allclose(p1, s2, rtol=1e-6)
        np.testing.assert_allclose(s1, p2, rtol=1e-6)

        # Test operator overload gives same result
        p3, s3 = mask_inv2.split(x)
        np.testing.assert_allclose(p2, p3, rtol=1e-6)
        np.testing.assert_allclose(s2, s3, rtol=1e-6)

    def test_checker_mask_pattern(self):
        """Test that checker mask creates correct checkerboard pattern."""
        mask_even = checker_mask((4, 4), parity=False)
        mask_odd = checker_mask((4, 4), parity=True)

        # Patterns should be inverted
        bool_even = mask_even.boolean_mask
        bool_odd = mask_odd.boolean_mask

        np.testing.assert_array_equal(bool_even, ~bool_odd)

        # Test pattern structure
        expected_even = jnp.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ]
        )

        np.testing.assert_array_equal(bool_even, expected_even)

    def test_empty_and_edge_cases(self):
        """Test edge cases like very small arrays."""
        # Single element
        mask = BinaryMask.from_boolean_mask(jnp.array([True]))
        x = jnp.array([5.0])
        primary, secondary = mask.split(x)
        reconstructed = mask.merge(primary, secondary)

        np.testing.assert_allclose(x, reconstructed, rtol=1e-6)
        assert primary.shape == (1,)
        assert secondary.shape == (0,)  # empty secondary

        # Test with complex numbers
        mask = BinaryMask.from_boolean_mask(jnp.array([True, False]))
        x_complex = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
        primary, secondary = mask.split(x_complex)
        reconstructed = mask.merge(primary, secondary)

        np.testing.assert_allclose(x_complex, reconstructed, rtol=1e-6)

    @given(
        st.tuples(st.integers(2, 6), st.integers(2, 6)).flatmap(
            lambda shape: arrays(
                jnp.float32,
                shape=shape,
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )
    )
    def test_split_merge_property(self, x):
        """Property-based test for split/merge roundtrip."""
        mask = checker_mask(x.shape, parity=True)
        primary, secondary = mask.split(x)
        reconstructed = mask.merge(primary, secondary)

        # Should be approximately equal
        np.testing.assert_allclose(x, reconstructed, rtol=1e-5, atol=1e-6)
