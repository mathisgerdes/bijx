"""
Shared testing utilities for bijx test suite.

This module provides reusable testing patterns, constants, and helper functions
that ensure consistency across all test files.

Also test MCMC samplers.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from bijx.bijections.base import Bijection

# Numerical stability constants (64-bit precision defaults)
# Global defaults for bijection round-trips and log-density checks
ATOL = 1e-9
RTOL = 1e-9

# Relaxed tolerance for numerical operations (e.g. fft, ODE)
ATOL_RELAXED = 1e-6
RTOL_RELAXED = 1e-6

# Global control for Hypothesis example count (overridable via env)
HYP_MAX_EXAMPLES = int(os.getenv("HYP_MAX_EXAMPLES", "10"))

# Domain ranges for different bijection types
# Keep small margins to avoid boundary pathologies in log-Jacobians.
GAUSSIAN_RANGE = (-3.0, 3.0)
UNIT_INTERVAL_RANGE = (1e-3, 1.0 - 1e-3)
MINUS_ONE_TO_ONE_RANGE = (-1.0 + 1e-3, 1.0 - 1e-3)
POSITIVE_RANGE = (1e-2, 10.0)

# Common array shapes for testing
TEST_SHAPES = [
    (),  # scalar
    (1,),  # 1D single element
    (3,),  # 1D small
    (2, 2),  # 2D square
    (3, 2),  # 2D rectangular
]


random_seeds = st.integers(min_value=0, max_value=2**32 - 1)


def is_valid_array(x: jnp.ndarray) -> bool:
    """
    Check if array contains valid values (no NaN or inf).

    Args:
        x: Input array to validate

    Returns:
        True if array contains only finite values, False otherwise
    """
    return not (jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)))


# Generic bounded arrays strategy
def _finite_floats(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@st.composite
def bounded_arrays(
    draw,
    min_value: float,
    max_value: float,
    shape: tuple | None = None,
    dtype=np.float64,
) -> jnp.ndarray:
    if shape is None:
        shape = draw(st.sampled_from(TEST_SHAPES))
    return draw(
        arrays(
            dtype,
            shape,
            elements=_finite_floats(min_value, max_value),
        )
    )


def check_inverse(bijection: Bijection, x: jnp.ndarray) -> None:
    """
    Check that reverse(forward(x)) ≈ x and log densities are consistent.

    This is the fundamental test for bijection correctness.

    Args:
        bijection: Bijection to test
        x: Input array

    Raises:
        AssertionError:
            If bijection produces invalid outputs or fails inverse consistency.
    """
    # Skip empty arrays - they don't provide meaningful test information
    if x.size == 0:
        return

    # Initialize log density to zero
    log_density = jnp.zeros(())

    # Forward then reverse
    y, ld_forward = bijection.forward(x, log_density)

    # Check forward output validity - this should not fail for valid inputs
    if not is_valid_array(y):
        raise AssertionError(f"Forward transformation produced invalid output: {y}")

    x_back, ld_back = bijection.reverse(y, ld_forward)

    # Check reverse output validity
    if not is_valid_array(x_back):
        raise AssertionError(
            f"Reverse transformation produced invalid output: {x_back}"
        )

    # Check x ≈ x_back
    np.testing.assert_allclose(
        x,
        x_back,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Inverse consistency failed: forward(reverse(x)) != x",
    )
    np.testing.assert_allclose(
        ld_back,
        0,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Log density not properly restored after round trip",
    )


def check_log_density(bijection: Bijection, x: jnp.ndarray) -> None:
    """
    Check that the log density change matches computation with jax.vjp.

    Validates that the bijection correctly computes log-determinant of Jacobian.

    Args:
        bijection: Bijection to test
        x: Input array

    Raises:
        AssertionError: If bijection produces invalid outputs or incorrect log density
    """
    # Skip empty arrays - they don't provide meaningful test information
    if x.size == 0:
        return

    # Use scalar input for consistent testing
    if hasattr(x, "shape") and len(x.shape) > 0:
        x_test = jnp.array(x.flatten()[0], dtype=jnp.float64)
    else:
        x_test = jnp.array(x, dtype=jnp.float64)

    # Define function that returns only the forward transformation
    def forward_fn(x_in: jnp.ndarray) -> jnp.ndarray:
        y, _ = bijection.forward(x_in, 0.0)
        return y

    # Compute the Jacobian using vector-Jacobian product
    y, vjp_fn = jax.vjp(forward_fn, x_test)

    # Check that forward transformation is valid
    if not is_valid_array(y):
        raise AssertionError(
            f"Forward transformation in log density test produced invalid output: {y}"
        )

    # For scalar operations, the Jacobian is the derivative
    jacobian = vjp_fn(jnp.ones_like(y))[0]

    # Check that Jacobian computation is valid
    if not is_valid_array(jacobian):
        raise AssertionError(
            f"Jacobian computation produced invalid result: {jacobian}"
        )

    log_det_jacobian = jnp.log(jnp.abs(jacobian))

    # Get the reported log density change from the bijection
    _, reported_log_density = bijection.forward(x_test, 0.0)

    # Check that computed values are valid
    if not is_valid_array(log_det_jacobian):
        raise AssertionError(
            f"Log determinant computation produced invalid result: {log_det_jacobian}"
        )

    if not is_valid_array(reported_log_density):
        raise AssertionError(
            f"Bijection reported invalid log density: {reported_log_density}"
        )

    # The negative of the reported log density should match the log determinant
    np.testing.assert_allclose(
        -reported_log_density,
        log_det_jacobian,
        atol=ATOL,
        rtol=RTOL,
        err_msg=(
            f"Log density mismatch: reported={reported_log_density}, "
            f"expected={-log_det_jacobian}"
        ),
    )


def summarize_array(x: jnp.ndarray) -> str:
    x = jnp.asarray(x)
    return (
        f"shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={jnp.min(x):.3g}, max={jnp.max(x):.3g}, mean={jnp.mean(x):.3g}"
    )


def check_bijection_all_safe(
    bijection: Bijection, x: jnp.ndarray, *, return_reason: bool = False
) -> tuple[bool, str] | bool:
    try:
        check_inverse(bijection, x)
        check_log_density(bijection, x)
        return (True, "") if return_reason else True
    except Exception as e:  # noqa: BLE001 — we want to capture for diagnostics
        reason = f"{type(e).__name__}: {e}; x[{summarize_array(x)}]"
        return (False, reason) if return_reason else False


# Hypothesis strategies for different domains
@st.composite
def gaussian_domain_inputs(draw, shape: tuple | None = None) -> jnp.ndarray:
    """Generate arrays of valid inputs for unbounded domains."""
    if shape is None:
        shape = draw(st.sampled_from(TEST_SHAPES))

    return draw(bounded_arrays(GAUSSIAN_RANGE[0], GAUSSIAN_RANGE[1], shape))


@st.composite
def unit_interval_inputs(draw, shape: tuple | None = None) -> jnp.ndarray:
    """Generate arrays of valid inputs for [0, 1] domain bijections."""
    if shape is None:
        shape = draw(
            st.sampled_from(TEST_SHAPES[:2])
        )  # Smaller shapes for bounded domains

    return draw(bounded_arrays(UNIT_INTERVAL_RANGE[0], UNIT_INTERVAL_RANGE[1], shape))


@st.composite
def minus_one_to_one_inputs(draw, shape: tuple | None = None) -> jnp.ndarray:
    """Generate arrays of valid inputs for [-1, 1] domain bijections."""
    if shape is None:
        shape = draw(
            st.sampled_from(TEST_SHAPES[:2])
        )  # Smaller shapes for bounded domains

    return draw(
        bounded_arrays(MINUS_ONE_TO_ONE_RANGE[0], MINUS_ONE_TO_ONE_RANGE[1], shape)
    )


@st.composite
def positive_inputs(draw, shape: tuple | None = None) -> jnp.ndarray:
    """Generate arrays of valid positive inputs for [0, ∞) domain bijections."""
    if shape is None:
        shape = draw(
            st.sampled_from(TEST_SHAPES[:2])
        )  # Smaller shapes for bounded domains

    return draw(bounded_arrays(POSITIVE_RANGE[0], POSITIVE_RANGE[1], shape))


@st.composite
def batch_shapes(draw) -> tuple:
    """Generate reasonable batch shapes for testing."""
    return draw(
        st.sampled_from(
            [
                (),  # no batch
                (1,),  # single batch
                (5,),  # small batch
                (2, 3),  # multi-dimensional batch
            ]
        )
    )


def assert_shapes_compatible(x: jnp.ndarray, y: jnp.ndarray, message: str = "") -> None:
    """Assert that two arrays have compatible shapes for testing."""
    if x.shape != y.shape:
        raise AssertionError(f"Shape mismatch: {x.shape} vs {y.shape}. {message}")


def assert_finite_and_real(x: jnp.ndarray, name: str = "array") -> None:
    """Assert that array contains only finite, real values."""
    if not is_valid_array(x):
        raise AssertionError(f"{name} contains NaN or inf values")

    if jnp.iscomplexobj(x):
        raise AssertionError(f"{name} has imaginary components")


# Fourier and array generation utilities
@st.composite
def real_space_shapes(draw) -> tuple[int, ...]:
    """Generate valid real-space shapes for FFT and lattice tests."""
    num_dims = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(draw(st.integers(min_value=2, max_value=5)) for _ in range(num_dims))
    return shape


@st.composite
def random_real_arrays(draw) -> jnp.ndarray:
    """Generate random real arrays with modest magnitudes for stability."""
    shape = draw(real_space_shapes())
    return draw(
        arrays(
            np.float32,
            shape,
            elements=st.floats(
                min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
