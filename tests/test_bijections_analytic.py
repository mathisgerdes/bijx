"""
Tests for analytic bijections in bijx.bijections.analytic.

This module tests:
- CubicRational: algebraic rational function bijection
- SinhConjugation: sinh/arcsinh conjugation-based bijection
- CubicConjugation: cubic polynomial conjugation-based bijection

All tests verify:
1. Inverse consistency: forward then reverse returns original input
2. Log density correctness: matches autodiff computation
3. Parameter constraints and numerical stability
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given
from hypothesis import strategies as st

from bijx.bijections.analytic import CubicConjugation, CubicRational, SinhConjugation

from .utils import ATOL, RTOL, check_inverse, check_log_density, gaussian_domain_inputs


def make_cubic_rational(loc=0.0, alpha=2.0, beta=1.0):
    """Helper to create CubicRational with no transforms."""
    return CubicRational(
        loc=nnx.Param(jnp.array(loc)),
        alpha=nnx.Param(jnp.array(alpha)),
        beta=nnx.Param(jnp.array(beta)),
        alpha_transform=None,
        beta_transform=None,
        loc_transform=None,
    )


def make_sinh(loc=0.0, alpha=1.0, beta=0.0, mu=0.0, nu=0.0):
    """Helper to create SinhConjugation with no transforms."""
    return SinhConjugation(
        loc=nnx.Param(jnp.array(loc)),
        alpha=nnx.Param(jnp.array(alpha)),
        beta=nnx.Param(jnp.array(beta)),
        mu=nnx.Param(jnp.array(mu)),
        nu=nnx.Param(jnp.array(nu)),
        alpha_transform=None,
    )


def make_cubic(loc=0.0, beta=0.0, a=1.0, b=1.0):
    """Helper to create CubicConjugation with no transforms."""
    return CubicConjugation(
        loc=nnx.Param(jnp.array(loc)),
        beta=nnx.Param(jnp.array(beta)),
        a=nnx.Param(jnp.array(a)),
        b=nnx.Param(jnp.array(b)),
        a_transform=None,
        b_transform=None,
    )


class TestCubicRational:
    """Tests for CubicRational with rational function transform."""

    @given(gaussian_domain_inputs(shape=()))
    def test_inverse_consistency(self, x):
        """Test that forward then reverse returns original input."""
        bijection = make_cubic_rational(loc=0.0, alpha=2.0, beta=1.0)
        check_inverse(bijection, x)

    @given(gaussian_domain_inputs(shape=()))
    def test_log_density_autodiff(self, x):
        """Test that log density change matches autodiff computation."""
        bijection = make_cubic_rational(loc=0.0, alpha=2.0, beta=1.0)
        check_log_density(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),  # loc
        st.floats(min_value=-0.9, max_value=7.9),  # alpha (within constraints)
        st.floats(min_value=0.1, max_value=5.0),  # beta (positive)
    )
    def test_parametric_inverse_consistency(self, x, loc, alpha, beta):
        """Test inverse consistency across parameter ranges."""
        bijection = make_cubic_rational(loc=loc, alpha=alpha, beta=beta)
        check_inverse(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),
        st.floats(min_value=-0.9, max_value=7.9),
        st.floats(min_value=0.1, max_value=5.0),
    )
    def test_parametric_log_density(self, x, loc, alpha, beta):
        """Test log density correctness across parameter ranges."""
        bijection = make_cubic_rational(loc=loc, alpha=alpha, beta=beta)
        check_log_density(bijection, x)

    def test_asymptotic_identity(self):
        """Test that bijection approaches identity as x → ±∞."""
        bijection = make_cubic_rational(loc=0.0, alpha=3.0, beta=1.0)

        # Test large positive values
        x_large_pos = jnp.array(100.0)
        y, _ = bijection.forward(x_large_pos, 0.0)
        np.testing.assert_allclose(
            y,
            x_large_pos,
            rtol=0.05,
            err_msg="Not asymptotically identity for large positive x",
        )

        # Test large negative values
        x_large_neg = jnp.array(-100.0)
        y, _ = bijection.forward(x_large_neg, 0.0)
        np.testing.assert_allclose(
            y,
            x_large_neg,
            rtol=0.05,
            err_msg="Not asymptotically identity for large negative x",
        )

    def test_parameter_transforms(self):
        """Test that parameter transforms enforce constraints."""
        rngs = nnx.Rngs(42)
        bijection = CubicRational(
            loc=nnx.Param(jnp.array(0.0)),
            alpha=nnx.Param(jnp.array(0.0)),  # Will be transformed
            beta=nnx.Param(jnp.array(0.0)),  # Will be transformed
            rngs=rngs,
        )

        # Alpha should be in (-1, 8) after transform
        alpha_val = bijection.alpha.get_value()
        assert -1 < alpha_val < 8, f"Alpha {alpha_val} not in valid range (-1, 8)"

        # Beta should be positive after transform
        beta_val = bijection.beta.get_value()
        assert beta_val > 0, f"Beta {beta_val} not positive"

    def test_jacobian_positivity(self):
        """Test that Jacobian determinant is always positive (bijection monotonic)."""
        bijection = make_cubic_rational(
            loc=0.0,
            alpha=3.0,
            beta=1.0,
        )

        x = jnp.linspace(-5, 5, 50)
        y, _ = bijection.forward(x, 0.0)
        log_jac = bijection.log_jac(x, y)

        # Log Jacobian should be finite
        assert jnp.all(jnp.isfinite(log_jac)), "Log Jacobian contains non-finite values"

        # Jacobian should be positive (log_jac should be real)
        # This is implicitly tested by being finite, but let's be explicit
        jac = jnp.exp(log_jac)
        assert jnp.all(jac > 0), "Jacobian determinant should be positive everywhere"


class TestSinhConjugation:
    """Tests for SinhConjugation with sinh/arcsinh conjugation."""

    @given(gaussian_domain_inputs(shape=()))
    def test_inverse_consistency(self, x):
        """Test that forward then reverse returns original input."""
        bijection = make_sinh(
            loc=0.0,
            alpha=1.0,
            beta=0.0,
            mu=0.0,
            nu=0.0,
        )
        check_inverse(bijection, x)

    @given(gaussian_domain_inputs(shape=()))
    def test_log_density_autodiff(self, x):
        """Test that log density change matches autodiff computation."""
        bijection = make_sinh(
            loc=0.0,
            alpha=1.0,
            beta=0.0,
            mu=0.0,
            nu=0.0,
        )
        check_log_density(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),  # loc
        st.floats(min_value=0.1, max_value=3.0),  # alpha (positive)
        st.floats(min_value=-3, max_value=3),  # beta
        st.floats(min_value=-1, max_value=1),  # mu
        st.floats(min_value=-1, max_value=1),  # nu
    )
    def test_parametric_inverse_consistency(self, x, loc, alpha, beta, mu, nu):
        """Test inverse consistency across parameter ranges."""
        bijection = make_sinh(
            loc=loc,
            alpha=alpha,
            beta=beta,
            mu=mu,
            nu=nu,
        )
        check_inverse(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),
        st.floats(min_value=0.1, max_value=3.0),
        st.floats(min_value=-3, max_value=3),
        st.floats(min_value=-1, max_value=1),
        st.floats(min_value=-1, max_value=1),
    )
    def test_parametric_log_density(self, x, loc, alpha, beta, mu, nu):
        """Test log density correctness across parameter ranges."""
        bijection = make_sinh(
            loc=loc,
            alpha=alpha,
            beta=beta,
            mu=mu,
            nu=nu,
        )
        check_log_density(bijection, x)

    def test_inverse_symmetry(self):
        """Test that inverse is obtained by parameter transformation."""
        loc, alpha, beta, mu, nu = 1.0, 2.0, 0.5, 0.3, -0.2

        bij_fwd = make_sinh(loc, alpha, beta, mu, nu)
        bij_inv = make_sinh(loc, alpha, -beta, -nu, -mu)

        x = jnp.linspace(-3, 3, 20)
        log_dens = jnp.zeros_like(x)

        # Forward with first bijection
        y, ld_fwd = bij_fwd.forward(x, log_dens)

        # "Reverse" with parameter-swapped bijection (should give same as reverse)
        x_back_param, ld_back_param = bij_inv.forward(y, ld_fwd)

        # Actual reverse
        x_back, ld_back = bij_fwd.reverse(y, ld_fwd)

        np.testing.assert_allclose(
            x_back_param,
            x_back,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Parameter-swapped forward doesn't match reverse",
        )
        np.testing.assert_allclose(
            ld_back_param,
            ld_back,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Log densities don't match for parameter-swapped bijection",
        )

    def test_numerical_stability_large_x(self):
        """Test numerical stability for large input values."""
        bijection = make_sinh(
            loc=0.0,
            alpha=1.0,
            beta=0.0,
            mu=0.0,
            nu=0.0,
        )

        # Test with large values where sinh could overflow
        x_large = jnp.array([10.0, 20.0, 50.0, -10.0, -20.0, -50.0])

        y, ld = bijection.forward(x_large, 0.0)
        x_back, ld_back = bijection.reverse(y, ld)

        # Check all values are finite
        assert jnp.all(
            jnp.isfinite(y)
        ), "Forward produced non-finite values for large x"
        assert jnp.all(jnp.isfinite(ld)), "Log density non-finite for large x"
        assert jnp.all(jnp.isfinite(x_back)), "Reverse produced non-finite values"

        # Check inverse consistency
        np.testing.assert_allclose(
            x_back,
            x_large,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Inverse consistency fails for large values",
        )

    def test_asymptotic_linearity(self):
        """Test that bijection is asymptotically linear for large |x|."""
        bijection = make_sinh(
            loc=0.0,
            alpha=1.0,
            beta=0.0,
            mu=0.0,
            nu=0.0,
        )

        # For large |x|, arcsinh(sinh(x)) ≈ x; transformation should be nearly identity
        x_large = jnp.array([100.0, -100.0])
        y, _ = bijection.forward(x_large, 0.0)

        # Should be approximately linear (y ≈ x + constant)
        slope = (y[0] - y[1]) / (x_large[0] - x_large[1])
        np.testing.assert_allclose(
            slope, 1.0, rtol=0.1, err_msg="Bijection not asymptotically linear"
        )


class TestCubicConjugation:
    """Tests for CubicConjugation with cubic polynomial conjugation."""

    @given(gaussian_domain_inputs(shape=()))
    def test_inverse_consistency(self, x):
        """Test that forward then reverse returns original input."""
        bijection = make_cubic(
            loc=0.0,
            beta=0.0,
            a=1.0,
            b=1.0,
        )
        check_inverse(bijection, x)

    @given(gaussian_domain_inputs(shape=()))
    def test_log_density_autodiff(self, x):
        """Test that log density change matches autodiff computation."""
        bijection = make_cubic(
            loc=0.0,
            beta=0.0,
            a=1.0,
            b=1.0,
        )
        check_log_density(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),  # loc
        st.floats(min_value=-3, max_value=3),  # beta
        st.floats(min_value=0.1, max_value=3.0),  # a (positive)
        st.floats(min_value=0.1, max_value=3.0),  # b (positive)
    )
    def test_parametric_inverse_consistency(self, x, loc, beta, a, b):
        """Test inverse consistency across parameter ranges."""
        bijection = make_cubic(
            loc=loc,
            beta=beta,
            a=a,
            b=b,
        )
        check_inverse(bijection, x)

    @given(
        gaussian_domain_inputs(shape=()),
        st.floats(min_value=-3, max_value=3),
        st.floats(min_value=-3, max_value=3),
        st.floats(min_value=0.1, max_value=3.0),
        st.floats(min_value=0.1, max_value=3.0),
    )
    def test_parametric_log_density(self, x, loc, beta, a, b):
        """Test log density correctness across parameter ranges."""
        bijection = make_cubic(
            loc=loc,
            beta=beta,
            a=a,
            b=b,
        )
        check_log_density(bijection, x)

    def test_inverse_by_beta_negation(self):
        """Test that inverse is obtained by negating beta."""
        loc, beta, a, b = 0.5, 1.0, 1.0, 0.5

        bij_fwd = make_cubic(loc, beta, a, b)
        bij_inv = make_cubic(loc, -beta, a, b)

        x = jnp.linspace(-3, 3, 20)
        log_dens = jnp.zeros_like(x)

        # Forward with first bijection
        y, ld_fwd = bij_fwd.forward(x, log_dens)

        # "Reverse" with negated beta
        x_back_neg, ld_back_neg = bij_inv.forward(y, ld_fwd)

        # Actual reverse
        x_back, ld_back = bij_fwd.reverse(y, ld_fwd)

        np.testing.assert_allclose(
            x_back_neg,
            x_back,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Beta-negated forward doesn't match reverse",
        )
        np.testing.assert_allclose(
            ld_back_neg,
            ld_back,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Log densities don't match for beta-negated bijection",
        )

    def test_cubic_solve_stability(self):
        """Test that cubic equation solver is numerically stable."""
        bijection = make_cubic(
            loc=0.0,
            beta=2.0,
            a=1.0,
            b=1.0,
        )

        # Test with various x values including edge cases
        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

        y, ld = bijection.forward(x, 0.0)
        x_back, _ = bijection.reverse(y, ld)

        # All values should be finite
        assert jnp.all(jnp.isfinite(y)), "Forward produced non-finite values"
        assert jnp.all(jnp.isfinite(x_back)), "Reverse produced non-finite values"

        # Should recover original
        np.testing.assert_allclose(
            x_back,
            x,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Cubic solver failed to invert accurately",
        )


class TestAnalyticBijectionComparison:
    """Comparative tests between the three analytic bijection types."""

    def test_all_identity_at_zero_params(self):
        """Test that all bijections reduce to identity with zero parameters."""
        x = jnp.linspace(-3, 3, 20)

        # CubicRational with alpha=0 should be identity
        bij_rat = make_cubic_rational(loc=0.0, alpha=0.0, beta=1.0)
        y_rat, _ = bij_rat.forward(x, 0.0)
        np.testing.assert_allclose(
            y_rat, x, atol=1e-7, err_msg="CubicRational not identity with alpha=0"
        )

        # Sinh with beta=mu=nu=0 should be identity
        bij_sinh = make_sinh(loc=0.0, alpha=1.0, beta=0.0, mu=0.0, nu=0.0)
        y_sinh, _ = bij_sinh.forward(x, 0.0)
        np.testing.assert_allclose(
            y_sinh, x, atol=1e-7, err_msg="Sinh not identity with zero params"
        )

        # Cubic with beta=0 should be identity
        bij_cubic = make_cubic(loc=0.0, beta=0.0, a=1.0, b=1.0)
        y_cubic, _ = bij_cubic.forward(x, 0.0)
        np.testing.assert_allclose(
            y_cubic, x, atol=1e-7, err_msg="Cubic not identity with beta=0"
        )

    def test_all_asymptotically_identity(self):
        """Test that all bijections approach identity for large |x|."""
        x_large = jnp.array([100.0, -100.0])

        bij_rat = make_cubic_rational(loc=0.0, alpha=3.0, beta=1.0)
        y_rat, _ = bij_rat.forward(x_large, 0.0)

        bij_sinh = make_sinh(loc=0.0, alpha=1.0, beta=1.0, mu=0.5, nu=0.5)
        y_sinh, _ = bij_sinh.forward(x_large, 0.0)

        bij_cubic = make_cubic(loc=0.0, beta=1.0, a=1.0, b=1.0)
        y_cubic, _ = bij_cubic.forward(x_large, 0.0)

        # All should be close to x for large |x|
        np.testing.assert_allclose(
            y_rat,
            x_large,
            rtol=0.1,
            err_msg="CubicRational not asymptotically identity",
        )
        np.testing.assert_allclose(
            y_sinh, x_large, rtol=0.2, err_msg="Sinh not asymptotically identity"
        )
        np.testing.assert_allclose(
            y_cubic, x_large, rtol=0.1, err_msg="Cubic not asymptotically identity"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
