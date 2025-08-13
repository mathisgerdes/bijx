"""
Tests for bijx probability distributions.

This module tests the core distribution functionality including sampling consistency,
log density evaluation accuracy, parameter gradient computation, and statistical
moment validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hypothesis import given
from hypothesis import strategies as st

from bijx import (
    ArrayDistribution,
    DiagonalGMM,
    Distribution,
    IndependentNormal,
    IndependentUniform,
)

from .utils import ATOL, RTOL, assert_finite_and_real, batch_shapes, random_seeds


class TestDistributionBase:
    """Tests for the base Distribution class interface."""

    def test_rng_handling_no_internal(self):
        """Test error handling when no RNG is provided."""

        # Create minimal subclass for testing
        class TestDist(Distribution):
            def get_batch_shape(self, x):
                return jnp.shape(x)

            def sample(self, batch_shape=(), rng=None):
                rng = self._get_rng(rng)
                return jnp.ones(batch_shape), jnp.zeros(batch_shape)

            def log_density(self, x):
                return jnp.zeros_like(x)

        dist = TestDist()
        with pytest.raises(ValueError, match="rngs must be provided"):
            dist.sample()

    def test_rng_handling_with_internal(self, rng_key):
        """Test RNG usage with internal rngs."""

        class TestDist(Distribution):
            def get_batch_shape(self, x):
                return jnp.shape(x)

            def sample(self, batch_shape=(), rng=None):
                rng = self._get_rng(rng)
                return jax.random.normal(rng, batch_shape), jnp.zeros(batch_shape)

            def log_density(self, x):
                return jnp.zeros_like(x)

        dist = TestDist(rngs=nnx.Rngs(rng_key))
        sample, log_p = dist.sample()
        assert_finite_and_real(sample, "sample")
        assert_finite_and_real(log_p, "log_p")

    def test_density_from_log_density(self):
        """Test that density() correctly exponentiates log_density()."""

        class TestDist(Distribution):
            def get_batch_shape(self, x):
                return jnp.shape(x)

            def sample(self, batch_shape=(), rng=None):
                return jnp.ones(batch_shape), jnp.zeros(batch_shape)

            def log_density(self, x):
                return -jnp.ones_like(x)  # log(exp(-1)) = -1

        dist = TestDist()
        x = jnp.array(1.0)

        log_p = dist.log_density(x)
        p = dist.density(x)

        expected_p = jnp.exp(log_p)
        np.testing.assert_allclose(p, expected_p, rtol=RTOL)
        np.testing.assert_allclose(p, jnp.exp(-1.0), rtol=RTOL)


class TestArrayDistribution:
    """Tests for the ArrayDistribution base class."""

    def test_event_shape_properties(self):
        """Test event shape property calculations."""
        event_shape = (3, 4, 5)
        dist = ArrayDistribution(event_shape=event_shape)

        assert dist.event_shape == event_shape
        assert dist.event_dim == 3
        assert dist.event_size == 60  # 3*4*5
        assert dist.event_axes == (-3, -2, -1)

    def test_batch_shape_extraction(self):
        """Test extraction of batch dimensions."""
        event_shape = (5,)
        dist = ArrayDistribution(event_shape=event_shape)

        # Test various batch shapes
        test_cases = [
            ((5,), ()),  # no batch
            ((1, 5), (1,)),  # single batch
            ((10, 5), (10,)),  # larger batch
            ((2, 3, 5), (2, 3)),  # multi-dimensional batch
        ]

        for array_shape, expected_batch in test_cases:
            x = jnp.zeros(array_shape)
            batch_shape = dist.get_batch_shape(x)
            assert batch_shape == expected_batch


class TestIndependentNormal:
    """Tests for IndependentNormal distribution."""

    @given(
        seed=random_seeds,
        event_shape=st.sampled_from([(), (1,), (3,), (2, 4), (3, 2, 5)]),
        batch_shape=batch_shapes(),
    )
    def test_sample_shapes(self, seed, event_shape, batch_shape):
        """Test that samples have correct shapes."""
        dist = IndependentNormal(event_shape=event_shape)

        samples, log_p = dist.sample(batch_shape=batch_shape, rng=jax.random.key(seed))

        expected_sample_shape = batch_shape + event_shape
        expected_log_p_shape = batch_shape

        assert samples.shape == expected_sample_shape
        assert log_p.shape == expected_log_p_shape
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

    @given(
        seed=random_seeds,
        event_shape=st.sampled_from([(), (1,), (5,), (2, 3)]),
    )
    def test_log_density_correctness(self, seed, event_shape):
        """Test log density computation matches standard normal."""
        dist = IndependentNormal(event_shape=event_shape)

        # Generate test samples
        samples, _ = dist.sample(batch_shape=(5,), rng=jax.random.key(seed))
        log_p = dist.log_density(samples)

        # Manual computation for independent normals
        expected_log_p = jnp.sum(
            jax.scipy.stats.norm.logpdf(samples),
            axis=tuple(range(-len(event_shape), 0)),
        )

        np.testing.assert_allclose(log_p, expected_log_p, atol=ATOL, rtol=RTOL)

    def test_sampling_consistency(self, rng_key):
        """Test that .sample log density matches .log_density."""
        event_shape = (10,)
        dist = IndependentNormal(event_shape=event_shape, rngs=nnx.Rngs(rng_key))

        samples, reported_log_p = dist.sample(batch_shape=(20,))
        computed_log_p = dist.log_density(samples)

        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)


class TestIndependentUniform:
    """Tests for IndependentUniform distribution."""

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        event_shape=st.sampled_from([(), (1,), (3,), (2, 4)]),
        batch_shape=batch_shapes(),
    )
    def test_sample_shapes_and_range(self, seed, event_shape, batch_shape):
        """Test sample shapes and range constraints."""
        dist = IndependentUniform(event_shape=event_shape)

        samples, log_p = dist.sample(batch_shape=batch_shape, rng=jax.random.key(seed))

        expected_sample_shape = batch_shape + event_shape
        expected_log_p_shape = batch_shape

        assert samples.shape == expected_sample_shape
        assert log_p.shape == expected_log_p_shape

        # All samples should be in [0, 1]
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 1.0)
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

    @given(
        seed=random_seeds,
        event_shape=st.sampled_from([(), (2,), (3, 2)]),
    )
    def test_log_density_correctness(self, seed, event_shape):
        """Test log density computation matches uniform distribution."""
        dist = IndependentUniform(event_shape=event_shape)

        # Test with valid samples in [0, 1]
        test_samples = jax.random.uniform(jax.random.key(seed), (5,) + event_shape)

        log_p = dist.log_density(test_samples)

        # Manual computation: log(1) = 0 for each dimension
        expected_log_p = jnp.zeros(5)

        np.testing.assert_allclose(log_p, expected_log_p, atol=ATOL, rtol=RTOL)

    def test_sampling_consistency(self, rng_key):
        """Test that reported log probabilities match actual densities."""
        event_shape = (4,)
        dist = IndependentUniform(event_shape=event_shape, rngs=nnx.Rngs(rng_key))

        samples, reported_log_p = dist.sample(batch_shape=(15,))
        computed_log_p = dist.log_density(samples)

        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)


class TestDiagonalGMM:
    """Tests for DiagonalGMM distribution."""

    def create_test_gmm(self, n_components=3, data_dim=2, seed=666):
        """Create a test GMM with reasonable parameters."""
        rngs = nnx.Rngs(seed)

        # Create means with appropriate dimensions
        means = (
            jax.random.normal(jax.random.key(seed), (n_components, data_dim)) * 2.0
        )  # Spread them out

        scales = jnp.ones((n_components, data_dim))
        weights = jnp.ones(n_components)  # Will be normalized by softmax

        return DiagonalGMM(means=means, scales=scales, weights=weights, rngs=rngs)

    def test_parameter_processing(self):
        """Test parameter processing functions work correctly."""
        gmm = self.create_test_gmm()

        # Check that weights are normalized
        weights = gmm.weights
        np.testing.assert_allclose(jnp.sum(weights), 1.0, rtol=RTOL)
        assert jnp.all(weights > 0)

        # Check scales are positive
        scales = gmm.scales
        assert jnp.all(scales > 0)

        # Check shapes
        assert gmm.means.shape == (3, 2)
        assert gmm.scales.shape == (3, 2)
        assert gmm.variances.shape == (3, 2)
        assert gmm.weights.shape == (3,)

    @given(batch_shape=batch_shapes())
    def test_sample_shapes(self, batch_shape):
        """Test GMM sampling produces correct shapes."""
        gmm = self.create_test_gmm(n_components=2, data_dim=3)

        samples, log_p = gmm.sample(batch_shape=batch_shape)

        expected_sample_shape = batch_shape + (3,)  # data_dim
        expected_log_p_shape = batch_shape

        assert samples.shape == expected_sample_shape
        assert log_p.shape == expected_log_p_shape
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

    def test_sampling_consistency(self):
        """Test that reported log probabilities match density evaluation."""
        gmm = self.create_test_gmm()

        samples, reported_log_p = gmm.sample(batch_shape=(50,))
        computed_log_p = gmm.log_density(samples)

        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)

    def test_log_density_evaluation(self, rng_key):
        """Test log density evaluation for known points."""
        # Simple single-component GMM at origin
        # rngs not required for deterministic eval-only test
        means = jnp.array([[0.0, 0.0]])
        scales = jnp.array([[1.0, 1.0]])
        weights = jnp.array([1.0])

        gmm = DiagonalGMM(
            means=means, scales=scales, weights=weights, rngs=nnx.Rngs(rng_key)
        )

        # Evaluate at origin - should match standard bivariate normal
        x = jnp.array([0.0, 0.0])
        log_p = gmm.log_density(x)

        # Expected: log(1/(2π)) = -log(2π)
        expected = -jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, atol=ATOL)

    def test_mixture_behavior(self):
        """Test that mixture has multiple modes as expected."""
        gmm = self.create_test_gmm(n_components=2, data_dim=1, seed=888)

        # Generate many samples to check for multimodality
        samples, _ = gmm.sample(batch_shape=(1000,))

        # With components at [0] and [2], we should see bimodal behavior
        # Check that we have samples in both regions
        left_samples = jnp.sum(samples < 1.0)
        right_samples = jnp.sum(samples > 1.0)

        # Both modes should have reasonable representation
        assert left_samples > 100
        assert right_samples > 100

    def test_single_component_equivalence(self, rng_key):
        """Test that single-component GMM behaves like a normal distribution."""
        # rngs not required for deterministic eval-only test
        mean = jnp.array([[1.0, 2.0]])
        scale = jnp.array([[0.5, 1.5]])
        weights = jnp.array([1.0])

        gmm = DiagonalGMM(
            means=mean, scales=scale, weights=weights, rngs=nnx.Rngs(rng_key)
        )

        # Test points
        test_x = jnp.array([1.0, 2.0])  # At the mean
        log_p_gmm = gmm.log_density(test_x)

        # Manual calculation for single multivariate normal
        diff = test_x - mean.squeeze()
        variances = scale.squeeze() ** 2
        mahalanobis = jnp.sum(diff**2 / variances)
        log_det = jnp.sum(jnp.log(variances))
        expected_log_p = -0.5 * (log_det + mahalanobis + 2 * jnp.log(2 * jnp.pi))

        np.testing.assert_allclose(log_p_gmm, expected_log_p, atol=ATOL)

    def test_batch_shape_handling(self):
        """Test that batch shapes are handled correctly."""
        gmm = self.create_test_gmm(data_dim=3)

        # Test single sample
        x_single = jnp.array([1.0, 1.0, 1.0])
        batch_shape_single = gmm.get_batch_shape(x_single)
        assert batch_shape_single == ()

        # Test batched samples
        x_batch = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_shape_batch = gmm.get_batch_shape(x_batch)
        assert batch_shape_batch == (2,)


class TestDistributionIntegration:
    """Integration tests across different distribution types."""

    def test_distribution_composition(self, rng_key):
        """Test that distributions can be used together in workflows."""
        # Create different distributions
        # Separate rng streams via fixture if needed;
        # here these modules are deterministic
        rngs1 = nnx.Rngs(0)
        rngs2 = nnx.Rngs(0)

        normal_dist = IndependentNormal(event_shape=(5,), rngs=rngs1)
        uniform_dist = IndependentUniform(event_shape=(3,), rngs=rngs2)

        # Sample from both
        normal_samples, normal_log_p = normal_dist.sample(batch_shape=(10,))
        uniform_samples, uniform_log_p = uniform_dist.sample(batch_shape=(10,))

        # Verify independent properties
        assert normal_samples.shape == (10, 5)
        assert uniform_samples.shape == (10, 3)
        assert_finite_and_real(normal_samples)
        assert_finite_and_real(uniform_samples)

        # Ranges should be different
        assert jnp.all(uniform_samples >= 0.0)
        assert jnp.all(uniform_samples <= 1.0)
        assert jnp.any(normal_samples < 0.0) or jnp.any(normal_samples > 1.0)

    def test_gradient_flow(self, rng_key):
        """Test that gradients flow correctly through distribution parameters."""

        def loss_fn(params, rng_key):
            gmm = DiagonalGMM(
                means=params["means"],
                scales=params["scales"],
                weights=params["weights"],
                rngs=nnx.Rngs(rng_key),
            )
            samples, log_p = gmm.sample(batch_shape=(5,))
            return -jnp.mean(log_p)  # Negative log likelihood

        # Initial parameters
        params = {
            "means": jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            "scales": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            "weights": jnp.array([0.5, 0.5]),
        }

        # Compute gradients
        loss_val, grads = jax.value_and_grad(lambda p: loss_fn(p, rng_key))(params)

        assert_finite_and_real(jnp.array(loss_val), "loss")
        for key, grad in grads.items():
            assert_finite_and_real(grad, f"gradient_{key}")
            assert grad.shape == params[key].shape
