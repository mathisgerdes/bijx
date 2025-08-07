r"""
Probability distributions and sampling utilities.

This module provides the core distribution interface and implementations for
common probability distributions used in normalizing flows. All distributions
support both sampling and log-density evaluation with automatic batch handling.
"""

from collections.abc import Callable

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import autovmap

from .utils import ShapeInfo, default_wrap


class Distribution(nnx.Module):
    """Base class for all probability distributions.

    Provides the fundamental interface for sampling and density evaluation
    that all distribution implementations must follow. Supports both explicit
    random key passing and automatic key management through the rngs attribute.

    The base class does not force a single event shape to be stored, which
    would be incompatible with general pytree objects that have different
    types of leaves. Instead, the `get_batch_shape` method must be implemented
    to extract the batch shape given a (batched or not) sample.

    The child class :class:`ArrayDistribution` should be used for simple
    distributions over arrays (not pytree objects).

    Args:
        rngs: Optional random number generator state for automatic key management.

    Note:
        Subclasses must implement get_batch_shape(), sample(), and log_density().
        The density() method is automatically derived from log_density().
    """

    def __init__(self, rngs: nnx.Rngs | None = None):
        self.rngs = rngs

    def _get_rng(self, rng: ftp.PRNGKey | None) -> ftp.PRNGKey:
        """Get random key from explicit argument or internal rngs.

        Raises:
            ValueError: If no rng provided and no internal rngs available.
        """
        if rng is None:
            if self.rngs is None:
                raise ValueError("rngs must be provided")
            rng = self.rngs.sample()
        return rng

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        """Extract batch dimensions from a sample.

        Args:
            x: A sample from this distribution.

        Returns:
            Tuple representing the batch dimensions of the sample.
        """
        raise NotImplementedError()

    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        """Generate samples from the distribution.

        Args:
            batch_shape: Shape of batch dimensions for vectorized sampling.
            rng: Random key for sampling, or None to use internal rngs.
            **kwargs: Additional distribution-specific sampling arguments.

        Returns:
            Tuple of (samples, log_densities) where samples have shape
            ``(*batch_shape, *event_shape)`` and log_densities have shape batch_shape.
        """
        raise NotImplementedError()

    def log_density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        """Evaluate log probability density at given points.

        Args:
            x: Points at which to evaluate density, with event dimensions
               matching the distribution's event shape.
            **kwargs: Additional distribution-specific evaluation arguments.

        Returns:
            Log density values with batch dimensions matching input.
        """
        raise NotImplementedError()

    def density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        """Evaluate probability density at given points.

        Args:
            x: Point(s) at which to evaluate density.
            **kwargs: Additional distribution-specific evaluation arguments.

        Returns:
            Density values (exponential of log density).
        """
        return jnp.exp(self.log_density(x, **kwargs))


class ArrayDistribution(Distribution):
    """Base class for distributions over multi-dimensional arrays.

    Extends the base Distribution class for distributions whose support
    consists of arrays with a fixed event shape. Provides utilities for
    handling event vs batch dimensions and shape manipulation.

    The event shape defines the dimensionality of individual samples,
    while batch dimensions allow vectorized operations over multiple samples.

    Args:
        event_shape: Shape of individual samples (event dimensions).
        rngs: Optional random number generator state.

    Example:
        >>> # 2D distribution (e.g., for images or lattice fields)
        >>> dist = SomeArrayDistribution(event_shape=(32, 32))
        >>> samples, log_p = dist.sample(batch_shape=(100,))  # 100 samples
        >>> assert samples.shape == (100, 32, 32)  # batch + event
        >>> assert log_p.shape == (100,)  # batch only
    """

    def __init__(self, event_shape: tuple[int, ...], rngs: nnx.Rngs | None = None):
        super().__init__(rngs)
        self.event_shape = event_shape
        self.shape_info = ShapeInfo(event_shape=event_shape)

    @property
    def event_dim(self):
        """Number of event dimensions."""
        return len(self.event_shape)

    @property
    def event_size(self):
        """Total number of elements in the event shape."""
        return np.prod(self.event_shape, dtype=int)

    @property
    def event_axes(self):
        """Axis indices corresponding to event dimensions."""
        return self.shape_info.event_axes

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        """Extract batch dimensions from an array sample."""
        return self.shape_info.process_event(x.shape)[0]


class IndependentNormal(ArrayDistribution):
    r"""Independent standard normal distribution over arrays.

    Each element of the array is independently distributed as a standard
    normal distribution $\mathcal{N}(0, 1)$. The total log density is the
    sum of individual element log densities.

    Example:
        >>> dist = IndependentNormal(event_shape=(10,), rngs=rngs)
        >>> x, log_p = dist.sample(batch_shape=(5,))
        >>> assert x.shape == (5, 10)
        >>> assert log_p.shape == (5,)
    """

    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        rng = self._get_rng(rng)
        x = jax.random.normal(rng, batch_shape + self.event_shape)
        return x, self.log_density(x)

    def log_density(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.norm.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class IndependentUniform(ArrayDistribution):
    r"""Independent uniform distribution over arrays on [0, 1].

    Each element of the array is independently distributed as a uniform
    distribution on the unit interval. The total log density is the sum
    of individual element log densities.

    Example:
        >>> dist = IndependentUniform(event_shape=(2, 3), rngs=rngs)
        >>> x, log_p = dist.sample(batch_shape=(100,))
        >>> assert jnp.all((x >= 0) & (x <= 1))  # All samples in [0,1]
    """

    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> tuple[jax.Array, jax.Array]:
        rng = self._get_rng(rng)
        x = jax.random.uniform(rng, batch_shape + self.event_shape)
        return x, self.log_density(x)

    def log_density(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.uniform.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class DiagonalGMM(Distribution):
    r"""Gaussian mixture model with diagonal covariance matrices.

    Implements a mixture of multivariate Gaussian distributions where each
    component has a diagonal covariance matrix. The mixture weights, means,
    and scales (standard deviations) can be learnable parameters.

    The log density is computed as:

    $$
    \log p(\mathbf{x}) =
    \log \sum_{k=1}^K w_k \exp\left(-\frac{1}{2}\sum_{i=1}^d
    \frac{(x_i - \mu_{k,i})^2}{\sigma_{k,i}^2} -
    \frac{1}{2}\sum_{i=1}^d \log(2\pi\sigma_{k,i}^2)\right)
    $$

    Args:
        means: Component means, shape (n_components, data_dim).
        scales: Component standard deviations, shape (n_components, data_dim).
        weights: Component mixture weights, shape (n_components,).
        process_scales: Function to ensure positive scales (default: abs).
        process_weights: Function to normalize weights (default: softmax).
        process_means: Function to process means (default: identity).
        rngs: Random number generator state.

    Note:
        Assumes data events are one-dimensional arrays (vectors).
    """

    def __init__(
        self,
        means,
        scales,
        weights,
        *,
        process_scales: Callable[[jax.Array], jax.Array] = jnp.abs,
        process_weights: Callable[[jax.Array], jax.Array] = jax.nn.softmax,
        process_means: Callable[[jax.Array], jax.Array] = lambda x: x,
        rngs: nnx.Rngs,
    ):
        super().__init__(rngs)

        self._means = default_wrap(means, init_fn=nnx.initializers.normal(), rngs=rngs)
        self._scales = default_wrap(scales, init_fn=nnx.initializers.ones, rngs=rngs)
        self._weights = default_wrap(weights, init_fn=nnx.initializers.ones, rngs=rngs)

        self.process_scales = process_scales
        self.process_weights = process_weights
        self.process_means = process_means

        assert (self.means.ndim, self.variances.ndim, self.weights.ndim) == (2, 2, 1)
        assert self.means.shape[0] == self.variances.shape[0] == self.weights.shape[0]

    def get_batch_shape(self, x: jax.Array) -> tuple[int, ...]:
        return x.shape[:-1]

    @property
    def means(self):
        return self.process_means(self._means.value)

    @property
    def variances(self):
        return self.scales**2

    @property
    def scales(self):
        return self.process_scales(self._scales.value)

    @property
    def weights(self):
        return self.process_weights(self._weights.value)

    @autovmap(x=1)
    def log_density(self, x):
        """Compute log probability density of the Gaussian mixture model.

        Args:
            x: Input points with shape (batch_size, data_dim).

        Returns:
            Log probability density for each point with shape (batch_size,).
        """
        weights = self.weights
        means = self.means
        variances = self.variances  # (components, dim)
        dim = x.size
        assert dim == means.shape[1]

        x_expanded = x[None, :]  # (components, dim)
        diff = x_expanded - means  # (components, dim)

        mahalanobis = jnp.sum(diff**2 / variances, axis=1)
        log_dets = jnp.sum(jnp.log(variances), axis=1)  # (components,)
        log_gaussians = -0.5 * (log_dets + mahalanobis + dim * jnp.log(2 * jnp.pi))

        log_gaussians = log_gaussians + jnp.log(weights)
        return jax.nn.logsumexp(log_gaussians, axis=0)

    def sample(self, batch_shape=(), rng=None):
        """Generate samples from the Gaussian mixture model.

        Args:
            batch_shape: Shape of batch dimensions for sampling.
            rng: Random key for sampling, or None to use internal rngs.

        Returns:
            Tuple of (samples, log_densities) where samples have shape
            (*batch_shape, data_dim) and log_densities have shape batch_shape.
        """
        count = int(np.prod(batch_shape) if batch_shape else 1)
        rng = self.rngs.sample() if rng is None else rng
        key_components, key_gaussian = jax.random.split(rng, 2)

        # Sample component indices based on weights
        component_indices = jax.random.choice(
            key_components,
            len(self.weights),
            shape=(count,),
            p=self.weights,
        )

        means = self.means
        scales = self.scales

        @jax.vmap
        def _sample(i, component):
            normal_samples = jax.random.normal(
                jax.random.fold_in(key_gaussian, i), means.shape[1:]
            )
            return means[component] + scales[component] * normal_samples

        samples = _sample(
            jnp.arange(count),
            component_indices,
        )

        log_density = self.log_density(samples)

        return samples.reshape(*batch_shape, -1), log_density.reshape(batch_shape)
