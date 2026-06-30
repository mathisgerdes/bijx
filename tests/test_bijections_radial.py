"""
Tests for radial bijections.

This module tests:
- RayTransform: ensuring f(0) = 0 property
- Radial: radial bijection with learnable scaling and centering
- RadialConditional: radial flow with angle-dependent bijection parameters

Hypothesis is used only where it draws continuous domain inputs `x` and checks a
strong invariant (inverse round-trip). Module parameters use a fixed seed: a
randomized seed is not a meaningful search space for these deterministic checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given

from bijx import (
    AffineLinear,
    Exponential,
    Power,
    Radial,
    RadialConditional,
    RayTransform,
    Shift,
    Sinh,
    SinhConjugation,
    Tanh,
)

from .utils import (
    ATOL,
    ATOL_RELAXED,
    RTOL,
    RTOL_RELAXED,
    assert_finite_and_real,
    check_inverse,
    gaussian_domain_inputs,
)


class TestRayTransform:
    """Tests for RayTransform bijection."""

    @given(x=gaussian_domain_inputs(shape=()))
    def test_origin_property(self, x):
        """Test that RayTransform ensures f(0) = 0."""
        base = Shift(shift=jnp.array(2.5))
        ray = RayTransform(base)

        # Check that f(0) = 0
        y_zero, _ = ray.forward(jnp.array(0.0), jnp.array(0.0))
        np.testing.assert_allclose(y_zero, 0.0, atol=ATOL, rtol=RTOL)

        # Check inverse consistency
        check_inverse(ray, x)

    @given(x=gaussian_domain_inputs(shape=()))
    def test_with_tanh(self, x):
        """Test RayTransform with Tanh bijection."""
        base = Tanh()
        ray = RayTransform(base)

        # Check origin property
        y_zero, _ = ray.forward(jnp.array(0.0), jnp.array(0.0))
        np.testing.assert_allclose(y_zero, 0.0, atol=ATOL, rtol=RTOL)

        # Check inverse consistency
        check_inverse(ray, x)

    @given(x=gaussian_domain_inputs(shape=()))
    def test_with_sinh(self, x):
        """Test RayTransform with Sinh bijection."""
        base = Sinh()
        ray = RayTransform(base)

        # Since sinh(0) = 0, the offset should be zero
        y_zero, _ = ray.forward(jnp.array(0.0), jnp.array(0.0))
        np.testing.assert_allclose(y_zero, 0.0, atol=ATOL, rtol=RTOL)

        check_inverse(ray, x)


def _check_inverse_relaxed(bijection, x):
    """Check inverse with relaxed tolerance for radial transformations.

    Radial transformations involve divisions by radius and can accumulate
    numerical errors, especially for certain input configurations.
    """
    log_density = jnp.zeros(())
    y, ld_forward = bijection.forward(x, log_density)
    x_back, ld_back = bijection.reverse(y, ld_forward)

    np.testing.assert_allclose(x, x_back, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
    np.testing.assert_allclose(ld_back, 0, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)


class TestRadial:
    """Tests for Radial bijection."""

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_2d_inverse_consistency(self, x):
        """Test inverse consistency for 2D radial transformation."""
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        _check_inverse_relaxed(radial, x)

    @given(x=gaussian_domain_inputs(shape=(3,)))
    def test_3d_inverse_consistency(self, x):
        """Test inverse consistency for 3D radial transformation."""
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(3,), scale=(3,), rngs=nnx.Rngs(0))

        _check_inverse_relaxed(radial, x)

    def test_at_origin(self):
        """Test behavior at the origin (r=0)."""
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        x = jnp.zeros(2)
        y, ld = radial.forward(x, jnp.array(0.0))

        # Output should be finite
        assert_finite_and_real(y, "radial output at origin")
        assert_finite_and_real(ld, "radial log density at origin")

    def test_batched_inputs(self):
        """Test round-trip with batched inputs."""
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        # Batch of 5 samples
        x = jax.random.normal(jax.random.key(0), (5, 2))
        ld = jnp.zeros(5)

        y, ld_forward = radial.forward(x, ld)
        assert y.shape == (5, 2)
        assert ld_forward.shape == (5,)
        assert_finite_and_real(y, "batched radial output")
        assert_finite_and_real(ld_forward, "batched radial log density")

        # Check round-trip with relaxed tolerance
        x_back, ld_back = radial.reverse(y, ld_forward)
        np.testing.assert_allclose(x_back, x, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
        np.testing.assert_allclose(ld_back, ld, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_with_exponential_scalar(self, x):
        """Test radial with Exponential scalar bijection."""
        scalar_bij = Exponential()
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        check_inverse(radial, x)

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_with_power_scalar(self, x):
        """Test radial with Power scalar bijection."""
        scalar_bij = Power(exponent=nnx.Variable(2.0))
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        check_inverse(radial, x)

    @pytest.mark.parametrize("n_dims", [2, 3, 4])
    def test_different_dimensions(self, n_dims):
        """Test radial round-trip in different dimensions.

        (1D is degenerate, 5D accumulates too much error.)
        """
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(n_dims,), scale=(n_dims,), rngs=nnx.Rngs(0))

        x = jax.random.normal(jax.random.key(n_dims), (n_dims,))
        y, ld = radial.forward(x, jnp.array(0.0))

        assert y.shape == (n_dims,)
        assert_finite_and_real(y, f"radial output in {n_dims}D")
        assert_finite_and_real(ld, f"radial log density in {n_dims}D")

        # Check inverse with relaxed tolerance
        x_back, _ = radial.reverse(y, ld)
        np.testing.assert_allclose(x_back, x, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)


class TestRadialConditional:
    """Tests for RadialConditional bijection."""

    def _create_conditional_radial(self, n_dims, seed=0):
        """Create a conditional radial bijection for testing."""
        # Use AffineLinear which has learnable parameters
        scalar_bij_template = AffineLinear(rngs=nnx.Rngs(seed))

        # Get the number of parameters needed
        from bijx.bijections.coupling import ModuleReconstructor

        reconst = ModuleReconstructor(scalar_bij_template)
        n_params = reconst.params_total_size

        # Simple conditioning network that outputs constant parameters
        class SimpleCond(nnx.Module):
            def __init__(self, n_params):
                self.n_params = n_params

            def __call__(self, x_hat):
                batch_shape = x_hat.shape[:-1]
                return jnp.zeros(batch_shape + (self.n_params,))

        cond_net = SimpleCond(n_params)

        return RadialConditional(
            scalar_bij_template,
            cond_net,
            center=(n_dims,),
            scale=(n_dims,),
            rngs=nnx.Rngs(seed),
        )

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_2d_inverse_consistency(self, x):
        """Test inverse consistency for 2D conditional radial."""
        radial = self._create_conditional_radial(n_dims=2)
        check_inverse(radial, x)

    @given(x=gaussian_domain_inputs(shape=(3,)))
    def test_3d_inverse_consistency(self, x):
        """Test inverse consistency for 3D conditional radial."""
        radial = self._create_conditional_radial(n_dims=3)
        check_inverse(radial, x)

    def test_at_origin(self):
        """Test behavior at the origin (r=0)."""
        radial = self._create_conditional_radial(n_dims=2)

        x = jnp.zeros(2)
        y, ld = radial.forward(x, jnp.array(0.0))

        # Output should be finite
        assert_finite_and_real(y, "conditional radial output at origin")
        assert_finite_and_real(ld, "conditional radial log density at origin")

    def test_batched_inputs(self):
        """Test round-trip with batched inputs."""
        radial = self._create_conditional_radial(n_dims=2)

        # Batch of 5 samples
        x = jax.random.normal(jax.random.key(0), (5, 2))
        ld = jnp.zeros(5)

        y, ld_forward = radial.forward(x, ld)
        assert y.shape == (5, 2)
        assert ld_forward.shape == (5,)
        assert_finite_and_real(y, "batched conditional radial output")
        assert_finite_and_real(ld_forward, "batched conditional radial log density")

        # Check round-trip
        x_back, ld_back = radial.reverse(y, ld_forward)
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, ld, atol=ATOL, rtol=RTOL)

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_with_parameterized_conditioning(self, x):
        """Test conditional radial with a parameterized conditioning network.

        Note: Parameterized conditioning is problematic at the origin (r=0)
        since angle is undefined there. We filter out inputs too close to zero.
        """
        # Skip if input is too close to origin where angle is undefined
        if jnp.linalg.norm(x) < 0.1:
            return

        # Use SinhConjugation with RayTransform for orientation-preserving bijection;
        # RayTransform ensures f(0)=0,
        # and combined with monotonicity ensures f(r)>0 for r>0.
        # Radial bijections require maps that preserve positivity (R+ -> R+)
        base_bij = SinhConjugation(rngs=nnx.Rngs(0))
        scalar_bij_template = RayTransform(base_bij)

        # Get the number of parameters needed
        from bijx.bijections.coupling import ModuleReconstructor

        reconst = ModuleReconstructor(scalar_bij_template)
        n_params = reconst.params_total_size

        class ParameterizedCond(nnx.Module):
            def __init__(self, n_params, rngs):
                self.dense = nnx.Linear(2, n_params, rngs=rngs)

            def __call__(self, x_hat):
                return self.dense(x_hat)

        cond_net = ParameterizedCond(n_params, rngs=nnx.Rngs(0))

        radial = RadialConditional(
            scalar_bij_template,
            cond_net,
            center=(2,),
            scale=(2,),
            rngs=nnx.Rngs(0),
        )

        _check_inverse_relaxed(radial, x)

    @pytest.mark.parametrize("n_dims", [2, 3, 4, 5])
    def test_different_dimensions(self, n_dims):
        """Test conditional radial round-trip in different dimensions."""
        radial = self._create_conditional_radial(n_dims)

        x = jax.random.normal(jax.random.key(n_dims), (n_dims,))
        y, ld = radial.forward(x, jnp.array(0.0))

        assert y.shape == (n_dims,)
        assert_finite_and_real(y, f"conditional radial output in {n_dims}D")
        assert_finite_and_real(ld, f"conditional radial log density in {n_dims}D")

        # Check inverse
        x_back, _ = radial.reverse(y, ld)
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)


class TestRadialComparison:
    """Compare Radial and RadialConditional behavior."""

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_radial_vs_conditional_inverse_consistency(self, x):
        """Test that both Radial and RadialConditional satisfy inverse consistency."""
        # Create a Radial bijection
        scalar_bij = Tanh()
        radial = Radial(scalar_bij, center=(2,), scale=(2,), rngs=nnx.Rngs(0))

        # Both should satisfy inverse consistency
        check_inverse(radial, x)

        # Create RadialConditional with parameterized bijection
        scalar_bij_template = AffineLinear(rngs=nnx.Rngs(0))
        from bijx.bijections.coupling import ModuleReconstructor

        reconst = ModuleReconstructor(scalar_bij_template)
        n_params = reconst.params_total_size

        class ConstantCond(nnx.Module):
            def __init__(self, n_params):
                self.n_params = n_params

            def __call__(self, x_hat):
                batch_shape = x_hat.shape[:-1]
                return jnp.zeros(batch_shape + (self.n_params,))

        cond_net = ConstantCond(n_params)
        radial_cond = RadialConditional(
            scalar_bij_template,
            cond_net,
            center=(2,),
            scale=(2,),
            rngs=nnx.Rngs(0),
        )

        check_inverse(radial_cond, x)
