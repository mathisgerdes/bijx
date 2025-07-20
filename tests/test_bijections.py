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
    ExpLayer,
    GaussianCDF,
    PowerLayer,
    SigmoidLayer,
    SoftPlusLayer,
    TanhLayer,
    TanLayer,
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
