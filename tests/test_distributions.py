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
    Const,
    DiagonalNormal,
    Distribution,
    GaussianMixture,
    IndependentNormal,
    IndependentUniform,
    MultivariateNormal,
)
from bijx.distributions import _cholesky_fold, _dim_to_chol_size

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


class TestMultivariateNormal:
    """Tests for MultivariateNormal distribution."""

    def create_test_mvn(self, dim=3, seed=42):
        """Create a test multivariate normal with reasonable parameters."""
        rngs = nnx.Rngs(seed)

        # Create mean and Cholesky factor
        mean = jnp.array([1.0, 2.0, 3.0][:dim]) if dim <= 3 else jnp.ones(dim)
        cholesky_size = _dim_to_chol_size(dim)
        cholesky = jnp.ones(cholesky_size) * 0.1  # Small values for numerical stability

        return MultivariateNormal(mean=mean, cholesky=cholesky, rngs=rngs)

    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        dim = 3
        mean = jnp.array([1.0, 2.0, 3.0])
        cholesky_size = _dim_to_chol_size(dim)
        cholesky = jnp.ones(cholesky_size) * 0.1

        mvn = MultivariateNormal(mean=mean, cholesky=cholesky)
        assert mvn.dim == dim
        assert mvn.mean.shape == (dim,)

    @given(
        seed=random_seeds,
        dim=st.sampled_from([2, 3]),
        batch_shape=batch_shapes(),
    )
    def test_sample_shapes(self, seed, dim, batch_shape):
        """Test that samples have correct shapes."""
        mvn = self.create_test_mvn(dim=dim, seed=seed)

        samples, log_p = mvn.sample(batch_shape=batch_shape, rng=jax.random.key(seed))

        expected_sample_shape = batch_shape + (dim,)
        expected_log_p_shape = batch_shape

        assert samples.shape == expected_sample_shape
        assert log_p.shape == expected_log_p_shape
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

    def test_sampling_consistency(self, rng_key):
        """Test that reported log probabilities match density evaluation."""
        mvn = self.create_test_mvn(dim=3, seed=42)

        samples, reported_log_p = mvn.sample(batch_shape=(5,), rng=rng_key)
        computed_log_p = mvn.log_density(samples)

        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)

    def test_log_density_correctness(self, rng_key):
        """Test log density matches analytical results for known cases."""
        # Test case: 2D standard normal
        mean = jnp.array([0.0, 0.0])
        cov = jnp.eye(2)
        cholesky_matrix = jnp.linalg.cholesky(cov)
        cholesky_vector = _cholesky_fold(cholesky_matrix)

        mvn = MultivariateNormal(
            mean=mean, cholesky=cholesky_vector, rngs=nnx.Rngs(rng_key)
        )

        # Test at origin
        x = jnp.array([0.0, 0.0])
        log_p = mvn.log_density(x)

        # Expected: log(1/(2π)) = -log(2π)
        expected = -jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, atol=ATOL)

        # Test at [1, 1]
        x = jnp.array([1.0, 1.0])
        log_p = mvn.log_density(x)

        # Expected: -1/2 * (1² + 1²) - log(2π) = -1 - log(2π)
        expected = -1.0 - jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, atol=ATOL)

    def test_log_density_vs_scipy(self, rng_key):
        """Test log density matches jnp.scipy.stats.multivariate_normal.logpdf."""
        # Test with a non-standard covariance matrix
        mean = jnp.array([1.0, -0.5])
        cov = jnp.array([[2.0, 0.8], [0.8, 1.5]])

        # Create our MultivariateNormal
        cholesky_vector = _cholesky_fold(jnp.linalg.cholesky(cov))
        mvn = MultivariateNormal(
            mean=mean, cholesky=cholesky_vector, rngs=nnx.Rngs(rng_key)
        )

        # Test points
        test_points = jnp.array(
            [
                [1.0, -0.5],  # at mean
                [0.0, 0.0],  # away from mean
            ]
        )

        for x in test_points:
            # Our implementation
            our_log_p = mvn.log_density(x)

            # JAX scipy reference
            scipy_log_p = jax.scipy.stats.multivariate_normal.logpdf(
                x, mean=mean, cov=cov
            )

            np.testing.assert_allclose(
                our_log_p,
                scipy_log_p,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"Mismatch at point {x}",
            )

    def test_given_dim_constructor(self):
        """Test the given_dim constructor."""
        dim = 4
        rngs = nnx.Rngs(123)

        mvn = MultivariateNormal.given_dim(dim=dim, rngs=rngs)
        assert mvn.dim == dim
        assert mvn.mean.shape == (dim,)

    def test_given_cov_constructor(self):
        """Test the given_cov constructor."""
        mean = jnp.array([1.0, -1.0])
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])

        mvn = MultivariateNormal.given_cov(mean=mean, cov=cov)
        assert mvn.dim == 2
        assert jnp.allclose(mvn.mean, mean, atol=ATOL)

        # Check that covariance reconstruction works
        reconstructed_cov = mvn.cov
        np.testing.assert_allclose(reconstructed_cov, cov, atol=ATOL)

    def test_cov_property(self):
        """Test that cov property reconstructs covariance correctly."""
        # Create a known covariance matrix
        cov_true = jnp.array([[4.0, 2.0], [2.0, 3.0]])

        mean = jnp.array([0.0, 0.0])
        mvn = MultivariateNormal.given_cov(mean=mean, cov=cov_true)

        reconstructed_cov = mvn.cov
        np.testing.assert_allclose(reconstructed_cov, cov_true, atol=ATOL)

    def test_gradient_flow(self, rng_key):
        """Test that gradients flow correctly through distribution parameters."""

        def loss_fn(cholesky_params, rng_key):
            mvn = MultivariateNormal(
                mean=jnp.array([0.0, 0.0]),
                cholesky=cholesky_params,
                rngs=nnx.Rngs(rng_key),
            )
            samples, log_p = mvn.sample(batch_shape=(2,))
            return -jnp.mean(log_p)  # Negative log likelihood

        # Initial Cholesky parameters (for 2D case)
        cholesky_size = _dim_to_chol_size(2)
        cholesky_params = jnp.ones(cholesky_size) * 0.1

        # Compute gradients
        loss_val, grads = jax.value_and_grad(lambda p: loss_fn(p, rng_key))(
            cholesky_params
        )

        assert_finite_and_real(jnp.array(loss_val), "loss")
        assert_finite_and_real(grads, "gradients")
        assert grads.shape == cholesky_params.shape

    def test_batch_shape_handling(self):
        """Test that batch shapes are handled correctly."""
        mvn = self.create_test_mvn(dim=3)

        # Test single sample
        x_single = jnp.array([1.0, 1.0, 1.0])
        batch_shape_single = mvn.get_batch_shape(x_single)
        assert batch_shape_single == ()

        # Test batched samples
        x_batch = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_shape_batch = mvn.get_batch_shape(x_batch)
        assert batch_shape_batch == (2,)


class TestDiagonalNormal:
    """Tests for DiagonalNormal distribution."""

    def create_test_diagonal_normal(self, dim=3, seed=42):
        """Create a test diagonal normal with reasonable parameters."""
        rngs = nnx.Rngs(seed)

        # Create mean and scales
        mean = jnp.array([1.0, 2.0, 3.0][:dim]) if dim <= 3 else jnp.ones(dim)
        scales = jnp.ones(dim) * 0.5  # Different scale for each dimension

        return DiagonalNormal(mean=mean, scales=scales, rngs=rngs)

    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        dim = 3
        mean = jnp.array([1.0, 2.0, 3.0])
        scales = jnp.array([0.5, 1.0, 1.5])

        dn = DiagonalNormal(mean=mean, scales=scales)
        assert dn.dim == dim
        assert dn.mean.shape == (dim,)
        assert dn.scales.shape == (dim,)

    @given(
        seed=random_seeds,
        dim=st.sampled_from([2, 3]),
        batch_shape=batch_shapes(),
    )
    def test_sample_shapes(self, seed, dim, batch_shape):
        """Test that samples have correct shapes."""
        dn = self.create_test_diagonal_normal(dim=dim, seed=seed)

        samples, log_p = dn.sample(batch_shape=batch_shape, rng=jax.random.key(seed))

        expected_sample_shape = batch_shape + (dim,)
        expected_log_p_shape = batch_shape

        assert samples.shape == expected_sample_shape
        assert log_p.shape == expected_log_p_shape
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

    def test_sampling_consistency(self, rng_key):
        """Test that reported log probabilities match density evaluation."""
        dn = self.create_test_diagonal_normal(dim=3, seed=42)

        samples, reported_log_p = dn.sample(batch_shape=(5,), rng=rng_key)
        computed_log_p = dn.log_density(samples)

        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)

    def test_log_density_correctness(self, rng_key):
        """Test log density matches analytical results for known cases."""
        # Test case: 2D diagonal normal with unit variance
        mean = jnp.array([0.0, 0.0])
        scales = jnp.array([1.0, 1.0])

        dn = DiagonalNormal(mean=mean, scales=scales, rngs=nnx.Rngs(rng_key))

        # Test at origin
        x = jnp.array([0.0, 0.0])
        log_p = dn.log_density(x)

        # Expected: log(1/(2π)) = -log(2π)
        expected = -jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, atol=ATOL)

        # Test at [1, 1]
        x = jnp.array([1.0, 1.0])
        log_p = dn.log_density(x)

        # Expected: -1/2 * (1² + 1²) - log(2π) = -1 - log(2π)
        expected = -1.0 - jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, atol=ATOL)

    def test_log_density_vs_scipy(self, rng_key):
        """Test log density matches jnp.scipy.stats.multivariate_normal.logpdf."""
        # Test with diagonal covariance
        mean = jnp.array([1.0, -0.5])
        scales = jnp.array([0.8, 1.2])
        cov = jnp.diag(scales**2)  # Diagonal covariance matrix

        # Create our DiagonalNormal
        dn = DiagonalNormal(mean=mean, scales=scales, rngs=nnx.Rngs(rng_key))

        # Test points
        test_points = jnp.array(
            [
                [1.0, -0.5],  # at mean
                [0.0, 0.0],  # away from mean
            ]
        )

        for x in test_points:
            # Our implementation
            our_log_p = dn.log_density(x)

            # JAX scipy reference
            scipy_log_p = jax.scipy.stats.multivariate_normal.logpdf(
                x, mean=mean, cov=cov
            )

            np.testing.assert_allclose(
                our_log_p,
                scipy_log_p,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"Mismatch at point {x}",
            )

    def test_given_dim_constructor(self):
        """Test the given_dim constructor."""
        dim = 4
        rngs = nnx.Rngs(123)

        dn = DiagonalNormal.given_dim(dim=dim, rngs=rngs)
        assert dn.dim == dim
        assert dn.mean.shape == (dim,)
        assert dn.scales.shape == (dim,)

    def test_variances_property(self):
        """Test that variances property returns scales squared."""
        scales = jnp.array([0.5, 1.0, 2.0])
        mean = jnp.array([0.0, 0.0, 0.0])

        dn = DiagonalNormal(mean=mean, scales=scales)

        expected_variances = scales**2
        np.testing.assert_allclose(dn.variances, expected_variances, atol=ATOL)

    def test_gradient_flow(self, rng_key):
        """Test that gradients flow correctly through distribution parameters."""

        def loss_fn(params, rng_key):
            dn = DiagonalNormal(
                mean=params[:2],
                scales=params[2:],
                rngs=nnx.Rngs(rng_key),
            )
            samples, log_p = dn.sample(batch_shape=(2,))
            return -jnp.mean(log_p)  # Negative log likelihood

        # Initial parameters (mean + scales concatenated)
        params = jnp.array([0.0, 0.0, 1.0, 1.0])

        # Compute gradients
        loss_val, grads = jax.value_and_grad(lambda p: loss_fn(p, rng_key))(params)

        assert_finite_and_real(jnp.array(loss_val), "loss")
        assert_finite_and_real(grads, "gradients")
        assert grads.shape == params.shape

    def test_batch_shape_handling(self):
        """Test that batch shapes are handled correctly."""
        dn = self.create_test_diagonal_normal(dim=3)

        # Test single sample
        x_single = jnp.array([1.0, 1.0, 1.0])
        batch_shape_single = dn.get_batch_shape(x_single)
        assert batch_shape_single == ()

        # Test batched samples
        x_batch = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_shape_batch = dn.get_batch_shape(x_batch)
        assert batch_shape_batch == (2,)

    def test_given_variances(self, rng_key):
        """Test the given_variances classmethod."""
        # Define mean and variances
        mean = jnp.array([1.0, 2.0, 3.0])
        variances = jnp.array(
            [4.0, 9.0, 16.0]
        )  # variances: 4, 9, 16 -> scales: 2, 3, 4

        # Create distribution with given variances
        dn = DiagonalNormal.given_variances(mean, variances, rngs=nnx.Rngs(rng_key))

        # Check that mean is correctly set
        np.testing.assert_allclose(dn.mean, mean, atol=ATOL)

        # Check that scales are correctly computed (sqrt of variances)
        expected_scales = jnp.sqrt(variances)
        np.testing.assert_allclose(dn.scales, expected_scales, atol=ATOL)

        # Test that the distribution produces valid samples
        samples, log_p = dn.sample(batch_shape=(10,), rng=rng_key)
        assert samples.shape == (10, 3)
        assert log_p.shape == (10,)
        assert_finite_and_real(samples, "samples")
        assert_finite_and_real(log_p, "log probabilities")

        # Test with nnx.Variable input to check type preservation
        variances_var = Const(jnp.array([1.0, 4.0, 9.0]))
        dn_var = DiagonalNormal.given_variances(
            mean, variances_var, rngs=nnx.Rngs(rng_key)
        )

        # Check that scales maintain the variable type
        assert isinstance(dn_var.scales_bare, Const)
        np.testing.assert_allclose(dn_var.scales, jnp.array([1.0, 2.0, 3.0]), atol=ATOL)


class TestGaussianMixture:
    """Tests for GaussianMixture distribution."""

    def create_test_gmm(self, n_components=3, data_dim=2, seed=42, diagonal=True):
        """Create a test Gaussian mixture model."""
        rngs = nnx.Rngs(seed)

        # Create means with appropriate dimensions
        means = jax.random.normal(jax.random.key(seed), (n_components, data_dim)) * 2.0

        if diagonal:
            # Diagonal covariances (variances)
            covariances = jnp.ones((n_components, data_dim)) * 0.5
        else:
            # Full covariances
            covariances = (
                jnp.eye(data_dim)[None, ...].repeat(n_components, axis=0) * 0.5
            )

        # Create weights
        weights = jnp.ones(n_components)  # Will be normalized by softmax

        return GaussianMixture(
            means=means, covariances=covariances, weights=weights, rngs=rngs
        )

    def test_diagonal_gmm_creation(self):
        """Test creating a diagonal Gaussian mixture model."""
        n_components = 3
        data_dim = 2
        gmm = self.create_test_gmm(
            n_components=n_components, data_dim=data_dim, diagonal=True
        )

        # Check shapes
        assert gmm.means.shape == (n_components, data_dim)
        assert gmm.covs.shape == (n_components, data_dim, data_dim)
        assert gmm.weights.shape == (n_components,)

        # Check that weights are normalized
        np.testing.assert_allclose(jnp.sum(gmm.weights), 1.0, rtol=RTOL)
        assert jnp.all(gmm.weights > 0)

        # Check covariances are positive (diagonal elements)
        covariances = gmm.covs
        diagonal_elements = jnp.diagonal(covariances, axis1=-2, axis2=-1)
        assert jnp.all(diagonal_elements > 0)

    def test_full_cov_gmm_creation(self):
        """Test creating a full covariance Gaussian mixture model."""
        n_components = 2
        data_dim = 3
        gmm = self.create_test_gmm(
            n_components=n_components, data_dim=data_dim, diagonal=False
        )

        # Check shapes
        assert gmm.means.shape == (n_components, data_dim)
        assert gmm.covs.shape == (n_components, data_dim, data_dim)
        assert gmm.weights.shape == (n_components,)

    @given(batch_shape=batch_shapes())
    def test_sample_shapes(self, batch_shape):
        """Test that samples have correct shapes."""
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

        samples, reported_log_p = gmm.sample(batch_shape=(10,))
        computed_log_p = gmm.log_density(samples)

        # Use relaxed tolerances for mixture models due to numerical precision
        np.testing.assert_allclose(reported_log_p, computed_log_p, atol=ATOL, rtol=RTOL)

    def test_log_density_evaluation(self, rng_key):
        """Test log density evaluation for known points."""
        # Simple single-component GMM at origin
        means = jnp.array([[0.0, 0.0]])
        covariances = jnp.array([[1.0, 1.0]])  # diagonal variances
        weights = jnp.array([1.0])

        gmm = GaussianMixture(
            means=means,
            covariances=covariances,
            weights=weights,
            rngs=nnx.Rngs(rng_key),
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

        # With components at different means, we should see multimodal behavior
        # Check that we have samples spread across the range
        sample_std = jnp.std(samples)
        sample_mean = jnp.mean(samples)

        # The spread should be reasonable (not collapsed to a single point)
        assert sample_std > 0.1
        assert jnp.abs(sample_mean) < 2.0  # Should be around the means we set

    def test_single_component_equivalence(self, rng_key):
        """Test that single-component GMM behaves like a normal distribution."""
        # Single component
        mean = jnp.array([[1.0, 2.0]])
        covariances = jnp.array([[0.5, 1.0]])  # diagonal variances
        weights = jnp.array([1.0])

        gmm = GaussianMixture(
            means=mean, covariances=covariances, weights=weights, rngs=nnx.Rngs(rng_key)
        )

        # Test point at the mean
        test_x = jnp.array([1.0, 2.0])
        log_p_gmm = gmm.log_density(test_x)

        # Manual calculation for single multivariate normal
        diff = test_x - mean.squeeze()
        variances = covariances.squeeze()
        mahalanobis = jnp.sum(diff**2 / variances)
        log_det = jnp.sum(jnp.log(variances))
        expected_log_p = -0.5 * (log_det + mahalanobis + 2 * jnp.log(2 * jnp.pi))

        np.testing.assert_allclose(log_p_gmm, expected_log_p, atol=ATOL)

    def test_batch_shape_handling(self):
        """Test that batch shapes are handled correctly."""
        gmm = self.create_test_gmm(n_components=1, data_dim=3)

        # Test single sample
        x_single = jnp.array([1.0, 1.0, 1.0])
        batch_shape_single = gmm.get_batch_shape(x_single)
        assert batch_shape_single == ()

        # Test batched samples
        x_batch = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        batch_shape_batch = gmm.get_batch_shape(x_batch)
        assert batch_shape_batch == (2,)

    def test_weights_none_default(self):
        """Test that weights=None creates uniform weights."""
        means = jnp.array([[0.0, 0.0], [2.0, 2.0]])
        covariances = jnp.array([[1.0, 1.0], [1.0, 1.0]])

        gmm = GaussianMixture(
            means=means, covariances=covariances, weights=None, rngs=nnx.Rngs(42)
        )

        # Should have uniform weights
        expected_weights = jnp.array([0.5, 0.5])
        np.testing.assert_allclose(gmm.weights, expected_weights, atol=ATOL)


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
            gmm = GaussianMixture(
                means=params["means"],
                covariances=params["covariances"],
                weights=params["weights"],
                rngs=nnx.Rngs(rng_key),
            )
            samples, log_p = gmm.sample(batch_shape=(5,))
            return -jnp.mean(log_p)  # Negative log likelihood

        # Initial parameters
        params = {
            "means": jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            "covariances": jnp.array([[1.0, 1.0], [1.0, 1.0]]),  # diagonal variances
            "weights": jnp.array([0.5, 0.5]),
        }

        # Compute gradients
        loss_val, grads = jax.value_and_grad(lambda p: loss_fn(p, rng_key))(params)

        assert_finite_and_real(jnp.array(loss_val), "loss")
        for key, grad in grads.items():
            assert_finite_and_real(grad, f"gradient_{key}")
            assert grad.shape == params[key].shape
