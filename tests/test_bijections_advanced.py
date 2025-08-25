"""
Tests for advanced bijections in bijx.

This module tests:
- Continuous normalizing flows (ContFlowDiffrax, ContFlowRK4, ContFlowCG)
- Automatic Jacobian vector fields (AutoJacVF)
- Rational quadratic splines (MonotoneRQSpline)
- Fourier space bijections (SpectrumScaling, FreeTheoryScaling)
- Advanced coupling layers with sophisticated parameter management
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given
from hypothesis import strategies as st

# Import bijections to test
from bijx import (
    AffineLinear,
    AutoJacVF,
    Chain,
    ContFlowDiffrax,
    ContFlowRK4,
    ConvVF,
    DiffraxConfig,
    FreeTheoryScaling,
    ModuleReconstructor,
    MonotoneRQSpline,
    SpectrumScaling,
    ToFourierData,
    rational_quadratic_spline,
)
from bijx.fourier import fft_momenta

# Import test utilities
from .utils import (
    ATOL,
    ATOL_RELAXED,
    RTOL,
    RTOL_RELAXED,
    assert_finite_and_real,
    check_inverse,
    check_log_density,
    gaussian_domain_inputs,
    unit_interval_inputs,
)


class TestContinuousFlows:
    """Tests for continuous normalizing flows."""

    @given(x=gaussian_domain_inputs(shape=()))
    def test_auto_jac_vf_scalar(self, x):
        """Test AutoJacVF for scalar vector fields (randomized x)."""

        def simple_vf(t, x):
            # Simple quadratic vector field: dx/dt = x^2 - 1
            return x**2 - 1.0

        # Wrap with automatic Jacobian computation
        auto_vf = AutoJacVF(simple_vf, event_dim=0)

        t = 0.5

        dx_dt, dlogp_dt = auto_vf(t, x)

        # Check shapes
        assert dx_dt.shape == x.shape
        assert dlogp_dt.shape == ()

        # sanity check output is what we defined it to be
        np.testing.assert_array_equal(dx_dt, x**2 - 1.0)

        # Check that dlogp_dt is -divergence (for scalar: -d/dx of dx_dt)
        # d/dx(x^2 - 1) = 2x, so -divergence = -2x
        expected_dlogp_dt = -2 * x
        np.testing.assert_allclose(dlogp_dt, expected_dlogp_dt, rtol=RTOL, atol=ATOL)

    @given(x=gaussian_domain_inputs(shape=(2,)))
    def test_auto_jac_vf_vector(self, x):
        """Test AutoJacVF for vector fields (randomized x)."""

        def spiral_vf(t, x):
            # 2D spiral vector field
            x1, x2 = x[..., 0], x[..., 1]
            dx1_dt = -x2 + 0.1 * x1
            dx2_dt = x1 + 0.1 * x2
            return jnp.stack([dx1_dt, dx2_dt], axis=-1)

        # Wrap with automatic Jacobian computation for vector field
        auto_vf = AutoJacVF(spiral_vf, event_dim=1)

        t = 0.0

        dx_dt, dlogp_dt = auto_vf(t, x)

        # Check shapes
        assert dx_dt.shape == x.shape
        assert dlogp_dt.shape == ()

        # Check that dx_dt matches the vector field definition
        expected_dx_dt = jnp.stack(
            [-x[..., 1] + 0.1 * x[..., 0], x[..., 0] + 0.1 * x[..., 1]], axis=-1
        )
        np.testing.assert_allclose(dx_dt, expected_dx_dt, rtol=RTOL, atol=ATOL)

        # For this particular spiral field, divergence = 0.1 + 0.1 = 0.2
        # So -divergence = -0.2
        expected_dlogp_dt = -0.2
        np.testing.assert_allclose(dlogp_dt, expected_dlogp_dt, rtol=RTOL, atol=ATOL)

    def test_cont_flow_rk4_basic(self):
        """Test ContFlowRK4 with simple vector field."""

        class LinearVF(nnx.Module):
            """Simple linear vector field for testing."""

            def __call__(self, t, x, **kwargs):
                # dx/dt = -0.5 * x (exponential decay)
                dx_dt = -0.5 * x
                # divergence of -0.5 * x is -0.5 for scalar case
                dlogp_dt = jnp.array(0.5)  # -(-0.5) = 0.5
                return dx_dt, dlogp_dt

        # Create RK4 continuous flow
        vf = LinearVF()
        flow = ContFlowRK4(vf, t_end=1.0, steps=10)  # dt=0.1 -> steps=10 for t_end=1.0

        # Test forward and inverse consistency
        x = jnp.array([2.0])
        log_density = jnp.array(0.0)

        # Check basic functionality
        y, ld_forward = flow.forward(x, log_density)
        assert_finite_and_real(y, "RK4 forward output")
        assert_finite_and_real(ld_forward, "RK4 forward log density")

        x_back, ld_back = flow.reverse(y, ld_forward)

        # Check inverse consistency
        # (with reasonable tolerance for numerical integration)
        np.testing.assert_allclose(x_back, x, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
        np.testing.assert_allclose(
            ld_back, log_density, atol=ATOL_RELAXED, rtol=RTOL_RELAXED
        )

    def test_conv_cnf_build_and_call(self, rng_key):
        """ConvCNF build test: shape handling and divergence shape."""
        cnf = ConvVF.build(
            kernel_shape=(3, 3), channel_shape=(1,), rngs=nnx.Rngs(rng_key)
        )
        x = jnp.ones((4, 4, 1))
        t = jnp.array(0.0)
        vel, neg_div = cnf(t, x)
        assert vel.shape == x.shape
        assert neg_div.shape == ()

    def test_cont_flow_diffrax_basic(self):
        """Test ContFlowDiffrax with simple vector field."""

        class LinearVF(nnx.Module):
            """Simple linear vector field for testing."""

            def __call__(self, t, x, **kwargs):
                # dx/dt = -0.2 * x (exponential decay)
                dx_dt = -0.2 * x
                # For scalar x, divergence is -0.2, so -divergence = 0.2
                dlogp_dt = jnp.array(0.2)
                return dx_dt, dlogp_dt

        # Create diffrax continuous flow
        vf = LinearVF()
        config = DiffraxConfig(
            t_start=0.0,
            t_end=2.0,
            dt=0.1,
            solver=diffrax.Heun(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        flow = ContFlowDiffrax(vf, config)

        # Test forward and inverse consistency
        x = jnp.array([1.5])
        log_density = jnp.array(0.0)

        # Check basic functionality
        y, ld_forward = flow.forward(x, log_density)
        assert_finite_and_real(y, "Diffrax forward output")
        assert_finite_and_real(ld_forward, "Diffrax forward log density")

        x_back, ld_back = flow.reverse(y, ld_forward)

        # Check inverse consistency
        # (with reasonable tolerance for numerical integration)
        np.testing.assert_allclose(x_back, x, atol=ATOL_RELAXED, rtol=RTOL_RELAXED)
        np.testing.assert_allclose(
            ld_back, log_density, atol=ATOL_RELAXED, rtol=RTOL_RELAXED
        )


class TestSplines:
    """Tests for rational quadratic splines."""

    def test_rational_quadratic_spline_basic(self):
        """Test basic rational quadratic spline functionality."""
        # Simple test with known parameters
        inputs = jnp.array([0.3, 0.7])

        # Create uniform bins (4 bins)
        n_bins = 4
        n_knots = n_bins - 1  # internal knots

        bin_widths = jnp.ones((2, n_bins))
        bin_heights = jnp.ones((2, n_bins))
        knot_slopes = jnp.ones((2, n_knots))

        # Test forward transformation
        outputs, log_det = rational_quadratic_spline(
            inputs, bin_widths, bin_heights, knot_slopes
        )

        assert_finite_and_real(outputs, "spline outputs")
        assert_finite_and_real(log_det, "spline log det")
        assert outputs.shape == inputs.shape
        assert log_det.shape == inputs.shape

        # Test inverse transformation
        inputs_back, log_det_inv = rational_quadratic_spline(
            outputs, bin_widths, bin_heights, knot_slopes, inverse=True
        )

        # Check inverse consistency
        np.testing.assert_allclose(inputs_back, inputs, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(log_det + log_det_inv, 0.0, atol=ATOL, rtol=RTOL)

    def test_spline_numerical_stability_boundaries(self):
        """Check stability for inputs near 0 and 1 and for extreme parameters."""
        eps = 1e-12
        x = jnp.array([0.0, eps, 1.0 - eps, 1.0])
        n_bins = 5
        n_knots = n_bins - 1
        big = 40.0
        small = -40.0
        widths = jnp.array([[big] * n_bins])
        heights = jnp.array([[small] * n_bins])
        slopes = jnp.array([[big] * n_knots])

        y, ld = rational_quadratic_spline(x, widths, heights, slopes)
        assert_finite_and_real(y, "spline forward near-boundary outputs")
        assert_finite_and_real(ld, "spline forward near-boundary log det")

        x2, ld2 = rational_quadratic_spline(y, widths, heights, slopes, inverse=True)
        assert_finite_and_real(x2, "spline inverse near-boundary outputs")
        assert_finite_and_real(ld2, "spline inverse near-boundary log det")

        np.testing.assert_allclose(x2, jnp.clip(x, 0, 1), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld + ld2, 0.0, atol=ATOL, rtol=RTOL)

    # Note: vectorized randomized param round-trip checked elsewhere

    def test_monotone_rq_spline_bijection(self, rng_key):
        """Test MonotoneRQSpline as a bijection."""
        # Create spline bijection
        n_knots = 6
        event_shape = ()  # scalar
        spline = MonotoneRQSpline(n_knots, event_shape, rngs=nnx.Rngs(rng_key))

        # Test with unit interval inputs
        x = jnp.array([0.2, 0.5, 0.8])
        log_density = jnp.zeros(3)

        # Check forward transformation
        y, ld_forward = spline.forward(x, log_density)
        assert_finite_and_real(y, "spline forward output")
        assert_finite_and_real(ld_forward, "spline forward log density")

        # Check inverse transformation
        x_back, ld_back = spline.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    @given(
        x=unit_interval_inputs(shape=(4, 3)),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_monotone_rq_spline_vectorized(self, x, seed):
        """Test MonotoneRQSpline with vectorized event shape (randomized batch)."""
        # Create spline for vector inputs
        n_knots = 5
        event_shape = (3,)  # 3D vector
        spline = MonotoneRQSpline(
            n_knots, event_shape, rngs=nnx.Rngs(jax.random.key(seed))
        )

        log_density = jnp.zeros(x.shape[0])

        # Test basic forward pass works (should not crash)
        y, ld_forward = spline.forward(x, log_density)
        assert_finite_and_real(y, "vectorized spline forward output")
        assert_finite_and_real(ld_forward, "vectorized spline forward log density")

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2,), (2, 2)])
    @pytest.mark.parametrize("knots", [4, 7])
    def test_monotone_rq_spline_vectorized_roundtrip(
        self, batch_shape, event_shape, knots, rng_key
    ):
        """Round-trip test across batch/event shape variants"""
        rngs = nnx.Rngs(rng_key)
        spline = MonotoneRQSpline(knots, event_shape, rngs=rngs)

        # Unit-interval inputs of shape batch + event
        if event_shape == ():
            x = jax.random.uniform(
                rngs(),
                shape=batch_shape or (),
                minval=0.05,
                maxval=0.95,
            )
            ld = jnp.zeros(batch_shape or ())
        else:
            x = jax.random.uniform(
                rngs(),
                shape=batch_shape + event_shape,
                minval=0.05,
                maxval=0.95,
            )
            ld = jnp.zeros(batch_shape)

        y, ld1 = spline.forward(x, ld)
        x2, ld2 = spline.reverse(y, ld1)

        assert_finite_and_real(y, "vectorized spline outputs")
        assert_finite_and_real(ld1, "vectorized spline log det forward")
        assert_finite_and_real(x2, "vectorized spline inverse outputs")
        assert_finite_and_real(ld2, "vectorized spline log det inverse")

        np.testing.assert_allclose(x2, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld2, ld, atol=ATOL, rtol=RTOL)

    @given(
        x=unit_interval_inputs(shape=()),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_spline_property_based(self, x, seed):
        """Property-based test for spline consistency."""
        spline = MonotoneRQSpline(
            4, (), rngs=nnx.Rngs(jax.random.key(seed))
        )  # Small spline for speed

        # Use safe checks for property-based testing with diagnostics
        check_inverse(spline, x)
        check_log_density(spline, x)


class TestFourierBijections:
    """Tests for Fourier space bijections."""

    def test_spectrum_scaling_basic(self):
        """Test basic SpectrumScaling functionality."""
        # Create simple 2D field for testing
        field_shape = (4, 4)
        x = jnp.ones(field_shape)
        log_density = jnp.array(0.0)

        # Create momentum-dependent scaling
        k_grid = fft_momenta(field_shape)
        k_squared = jnp.sum(k_grid**2, axis=-1)
        scaling = jnp.exp(-0.1 * k_squared)  # Exponential damping

        bijection = SpectrumScaling(scaling, channel_dim=0)

        # Test forward transformation
        y, ld_forward = bijection.forward(x, log_density)
        assert y.shape == x.shape
        assert_finite_and_real(y, "spectrum scaling output")
        assert_finite_and_real(ld_forward, "spectrum scaling log density")

        # Test inverse transformation
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    def test_free_theory_scaling(self, rng_key):
        """Test FreeTheoryScaling for physics applications."""
        # Create lattice field
        lattice_shape = (8, 8)
        x = jax.random.normal(rng_key, lattice_shape)
        log_density = jnp.array(0.0)

        # Create free theory scaling (mass term)
        mass_squared = 0.5
        bijection = FreeTheoryScaling(mass_squared, lattice_shape, channel_dim=0)

        # Test basic functionality
        y, ld_forward = bijection.forward(x, log_density)
        assert y.shape == x.shape
        assert_finite_and_real(y, "free theory output")
        assert_finite_and_real(ld_forward, "free theory log density")

        # Test inverse transformation
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    def test_to_fourier_data_bijection(self, rng_key):
        """Test ToFourierData conversion bijection."""
        # Create real field
        field_shape = (6, 6)
        x = jax.random.normal(rng_key, field_shape)
        log_density = jnp.array(0.0)

        from bijx.fourier import FFTRep

        bijection = ToFourierData(field_shape, rep=FFTRep.rfft)

        # Test forward (real -> Fourier)
        y, ld_forward = bijection.forward(x, log_density)
        # Output is FourierData object, check its .data attribute
        assert hasattr(y, "data"), "ToFourierData should return FourierData object"
        # FFT output is complex, so just check it's finite
        assert not jnp.any(
            jnp.isnan(y.data)
        ), "ToFourierData forward output contains NaN"
        assert not jnp.any(
            jnp.isinf(y.data)
        ), "ToFourierData forward output contains inf"
        assert_finite_and_real(ld_forward, "ToFourierData forward log density")

        # Test inverse (Fourier -> real)
        x_back, ld_back = bijection.reverse(y, ld_forward)

        # Check inverse consistency
        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)


class TestAdvancedCoupling:
    """Tests for advanced coupling layer functionality."""

    def test_module_reconstructor_basic(self, rng_key):
        """Test ModuleReconstructor parameter extraction."""
        # Create a bijection to extract parameters from
        bijection = AffineLinear(rngs=nnx.Rngs(rng_key))
        template = ModuleReconstructor(bijection)

        # Check parameter extraction
        assert template.params_total_size > 0
        assert len(template.params_shapes) > 0
        assert isinstance(template.params_dict, dict)
        assert not template.has_complex_params

        # Test reconstruction from array
        param_array = jnp.zeros(template.params_total_size)
        reconstructed = template.from_params(param_array)

        # Should be able to use the reconstructed bijection
        x = jnp.array([1.0])
        y, ld = reconstructed.forward(x, jnp.array(0.0))
        assert_finite_and_real(y, "reconstructed bijection output")
        assert_finite_and_real(ld, "reconstructed bijection log density")

    def test_module_reconstructor_with_spline(self, rng_key):
        """Test ModuleReconstructor with more complex bijection."""
        # Create spline bijection
        spline = MonotoneRQSpline(5, (), rngs=nnx.Rngs(rng_key))
        template = ModuleReconstructor(spline)

        # Test different parameter representations
        rng_key, k_arr = jax.random.split(rng_key)
        param_array = jax.random.normal(k_arr, (template.params_total_size,))

        # Test array reconstruction
        reconstructed_from_array = template.from_params(param_array)

        # Test dict reconstruction
        keys = jax.random.split(rng_key, len(template.params_shape_dict))
        param_dict = {
            key: jax.random.normal(k, shape)
            for (key, shape), k in zip(template.params_shape_dict.items(), keys)
        }
        reconstructed_from_dict = template.from_params(param_dict)

        # Test list reconstruction
        keys = jax.random.split(rng_key, len(template.params_shapes))
        param_list = [
            jax.random.normal(k, shape)
            for shape, k in zip(template.params_shapes, keys)
        ]
        reconstructed_from_list = template.from_params(param_list)

        # All should work for basic forward pass
        x = jnp.array([0.5])
        log_density = jnp.array(0.0)

        for reconstructed in [
            reconstructed_from_array,
            reconstructed_from_dict,
            reconstructed_from_list,
        ]:
            y, ld = reconstructed.forward(x, log_density)
            assert_finite_and_real(y, "reconstructed forward output")
            assert_finite_and_real(ld, "reconstructed forward log density")


class TestAdvancedIntegration:
    """Integration tests for advanced bijection combinations."""

    def test_fourier_spline_chain(self, rng_key):
        """Test chaining Fourier and spline bijections."""
        field_shape = (4, 4)
        rngs = nnx.Rngs(rng_key)

        spline = MonotoneRQSpline(8, field_shape, rngs=rngs)

        # Create chain: Fourier scaling -> flatten -> spline
        k_grid = fft_momenta(field_shape)
        k_squared = jnp.sum(k_grid**2, axis=-1)
        scaling = jnp.exp(-0.05 * k_squared)

        # Note: This is a conceptual test - actual chaining of these specific
        # bijections might require shape adjustments in practice
        spectrum_bij = SpectrumScaling(scaling, channel_dim=0)

        chain = Chain(
            spline,
            spectrum_bij,
        )

        x = jax.random.normal(rngs(), field_shape)
        log_density = jnp.array(0.0)

        # Test basic forward pass works
        y, ld_forward = chain.forward(x, log_density)
        assert_finite_and_real(y, "fourier chain forward output")
        assert_finite_and_real(ld_forward, "fourier chain forward log density")

    def test_advanced_bijection_gradient_flow(self):
        """Test gradient flow through advanced bijections."""

        def loss_fn(mass_param):
            # Create bijection with trainable parameter
            bijection = FreeTheoryScaling(mass_param, (4, 4), channel_dim=0)

            # Simple loss: transform a field and compute squared norm
            field = jnp.ones((4, 4))
            y, ld = bijection.forward(field, jnp.array(0.0))
            return jnp.sum(y**2) - ld

        mass_param = jnp.array(0.5)

        # Compute gradient
        loss_val, grad = jax.value_and_grad(loss_fn)(mass_param)

        assert_finite_and_real(jnp.array(loss_val), "advanced bijection loss")
        assert_finite_and_real(grad, "advanced bijection gradient")
        assert grad.shape == mass_param.shape

    def test_continuous_flow_batching(self):
        """Test continuous flow with batched inputs."""

        class SimpleBatchVF(nnx.Module):
            """Vector field that handles batched inputs."""

            def __call__(self, t, x, **kwargs):
                # Simple linear decay: dx/dt = -0.3 * x
                dx_dt = -0.3 * x
                # For batched inputs, return appropriate log density shape
                batch_shape = x.shape[:-1] if x.ndim > 0 else ()
                dlogp_dt = jnp.full(batch_shape, 0.3)
                return dx_dt, dlogp_dt

        vf = SimpleBatchVF()
        config = DiffraxConfig(
            t_start=0.0,
            t_end=1.0,
            dt=0.2,
            solver=diffrax.Euler(),
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        flow = ContFlowDiffrax(vf, config)

        # Test with batched inputs
        batch_size = 3
        x = jnp.ones((batch_size, 2))  # batch of 2D vectors
        log_density = jnp.zeros(batch_size)

        # Test forward
        y, ld_forward = flow.forward(x, log_density)
        assert y.shape == x.shape
        assert ld_forward.shape == log_density.shape
        assert_finite_and_real(y, "batched flow forward output")
        assert_finite_and_real(ld_forward, "batched flow forward log density")
