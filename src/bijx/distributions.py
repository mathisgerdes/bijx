r"""
Probability distributions and sampling utilities.

This module provides the core distribution interface and implementations for
common probability distributions used in normalizing flows. All distributions
support both sampling and log-density evaluation with automatic batch handling.
"""

from functools import partial

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import autovmap

from .utils import Const, ParamSpec, ShapeInfo, default_wrap

__all__ = [
    "Distribution",
    "ArrayDistribution",
    "IndependentNormal",
    "IndependentUniform",
    "MultivariateNormal",
    "DiagonalNormal",
    "MixtureStack",
    "GaussianMixture",
]


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
            ``(*batch_shape, *event_shape)``
            and log_densities have shape ``batch_shape``.
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


def _cov_init(key, shape):
    assert shape[-1] == shape[-2], "covariance must be a square matrix"
    cov = jnp.eye(shape[-1])
    noise = jax.random.normal(key, shape) * 0.005 / jnp.sqrt(shape[-1])
    return cov + (noise + noise.swapaxes(-1, -2))


def _cholesky_fold(cholesky_matrix):
    return cholesky_matrix[jnp.tril_indices(cholesky_matrix.shape[0])]


def _chol_size_to_dim(size):
    return int((np.sqrt(1 + 8 * size) - 1) / 2)


def _dim_to_chol_size(dim):
    return (dim * (dim + 1)) // 2


@autovmap(cholesky=1)
def _cholesky_unfold(cholesky):
    assert cholesky.ndim == 1
    size = _chol_size_to_dim(cholesky.size)
    cholesky_matrix = jnp.zeros((size, size))
    cholesky_matrix = cholesky_matrix.at[jnp.tril_indices(size)].set(cholesky)
    return cholesky_matrix


@autovmap(cov=2)
def _cov_to_cholesky(cov):
    return _cholesky_fold(jnp.linalg.cholesky(cov))


def _init_cholesky(key, shape):
    size = _chol_size_to_dim(shape[-1])
    cov = _cov_init(key, (*shape[:-1], size, size))
    return _cov_to_cholesky(cov)


class MultivariateNormal(ArrayDistribution):
    r"""Multivariate normal distribution with Cholesky parametrization.

    Implements a multivariate Gaussian distribution using Cholesky decomposition
    for numerical stability. The covariance matrix is represented by its Cholesky
    factor, which ensures positive definiteness and enables efficient sampling
    and density evaluation.

    The log density is computed as:

    $$
    \log p(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T
    \mathbf{L}^{-T}\mathbf{L}^{-1}(\mathbf{x} - \boldsymbol{\mu}) -
    \frac{d}{2}\log(2\pi) - \sum_{i=1}^d \log L_{ii}
    $$

    where $\mathbf{L}$ is the Cholesky factor such that
    $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$.

    Args:
        mean: Mean vector, shape (dim,) or scalar dim.
        cholesky: Cholesky factor vector (packed lower triangular),
            shape (dim*(dim+1)//2,).
        rngs: Optional random number generator state.
        var_cls: Variable class for parameters (default: nnx.Param).

    Example:
        >>> # 2D multivariate normal
        >>> mean = jnp.array([1.0, 2.0])
        >>> cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> dist = MultivariateNormal.given_cov(mean, cov)
        >>> samples, log_p = dist.sample(batch_shape=(100,), rng=rng)
        >>> assert samples.shape == (100, 2)
        >>> assert log_p.shape == (100,)

    The parameters an also be instantiated given shapes, and mean is sufficient:
        >>> dist = MultivariateNormal((3,), rngs=nnx.Rngs(0))  # 3D multivariate normal
    """

    def __init__(
        self,
        mean: ParamSpec,
        cholesky: ParamSpec | None = None,
        *,
        rngs=None,
        var_cls=nnx.Param,
        epsilon: float = 1e-10,
    ):
        """Initialize multivariate normal distribution.

        Args:
            mean: Mean vector specification.
            cholesky: Cholesky factor specification (packed lower triangular).
            rngs: Optional random number generator state.
            var_cls: Variable class for parameters.
            epsilon: Small regularization constant to ensure numerical stability.
        """
        self.epsilon = epsilon
        self.mean = default_wrap(
            mean, cls=var_cls, rngs=rngs, init_fn=nnx.initializers.normal()
        )
        dim = self.mean.shape[-1]

        if cholesky is None:
            cholesky = (*self.mean.shape[:-1], _dim_to_chol_size(dim))

        self.cholesky = default_wrap(
            cholesky, cls=var_cls, rngs=rngs, init_fn=_init_cholesky
        )
        assert dim == _chol_size_to_dim(self.cholesky.shape[-1]), (
            f"mean dimension {dim} does not match covariance dimension "
            + f"{_chol_size_to_dim(self.cholesky.shape[-1])} implicit in cholesky "
            + f"shape {self.cholesky.shape}"
        )
        super().__init__(rngs=rngs, event_shape=(dim,))

    @property
    def dim(self):
        """Dimensionality of the distribution."""
        return self.mean.size

    @property
    def cov(self):
        """Reconstruct covariance matrix from Cholesky factor."""
        matrix = _cholesky_unfold(self.cholesky.value)
        return jnp.einsum("...ij,...kj->...ik", matrix, matrix)

    @classmethod
    def given_dim(
        cls, dim: int, *, rngs: nnx.Rngs, var_cls=nnx.Param, epsilon: float = 1e-10
    ):
        """Create multivariate normal with given dimensionality.

        Args:
            dim: Dimensionality of the distribution.
            rngs: Random number generator state.
            var_cls: Variable class for parameters.
            epsilon: Small regularization constant to ensure numerical stability.

        Returns:
            MultivariateNormal instance.
        """
        cholesky_size = _dim_to_chol_size(dim)
        return cls(
            (dim,), (cholesky_size,), rngs=rngs, var_cls=var_cls, epsilon=epsilon
        )

    @classmethod
    def given_cov(
        cls,
        mean: ParamSpec,
        cov: ParamSpec,
        *,
        rngs=None,
        var_cls=Const,
        epsilon: float = 1e-10,
    ):
        """Create multivariate normal with given mean and covariance.

        Args:
            mean: Mean vector.
            cov: Covariance matrix.
            rngs: Optional random number generator state.
            var_cls: Variable class for parameters (default: Const).
            epsilon: Small regularization constant to ensure numerical stability.

        Returns:
            MultivariateNormal instance.
        """
        if not hasattr(cov, "shape"):
            cov = _cov_init(rngs.params(), cov)
        cholesky = _cov_to_cholesky(cov)
        return cls(mean, cholesky, rngs=rngs, var_cls=var_cls, epsilon=epsilon)

    def log_density(self, x):
        """Compute log probability density at given points.

        Args:
            x: Points at which to evaluate density, shape (..., dim).

        Returns:
            Log density values with batch dimensions matching input.
        """
        cholesky = _cholesky_unfold(self.cholesky.value)
        whitened = jnp.vectorize(
            partial(jax.lax.linalg.triangular_solve, lower=True, transpose_a=True),
            signature="(n,n),(n)->(n)",
        )(cholesky, x - self.mean)
        log_density = -1 / 2 * jnp.einsum("...i,...i->...", whitened, whitened)
        log_density -= self.dim / 2 * jnp.log(2 * jnp.pi)
        # Apply epsilon regularization to prevent NaN from log(0) or log(negative)
        safe_diagonal = jnp.abs(cholesky.diagonal()) + self.epsilon
        return log_density - jnp.log(safe_diagonal).sum(-1)

    def sample(self, batch_shape=(), rng=None):
        """Generate samples from the distribution.

        Args:
            batch_shape: Shape of batch dimensions for vectorized sampling.
            rng: Random key for sampling, or None to use internal rngs.

        Returns:
            Tuple of (samples, log_densities) where samples have shape
            ``(*batch_shape, dim)`` and log_densities have shape ``batch_shape``.
        """
        rng = self._get_rng(rng)
        noise = jax.random.normal(rng, (*batch_shape, self.dim))
        cholesky = _cholesky_unfold(self.cholesky.value)
        # Apply epsilon regularization to ensure valid Cholesky decomposition
        safe_diagonal = jnp.abs(cholesky.diagonal()) + self.epsilon
        cholesky = cholesky.at[jnp.diag_indices(cholesky.shape[-1])].set(safe_diagonal)
        samples = self.mean + jnp.einsum("...ij,...j->...i", cholesky, noise)
        return samples, self.log_density(samples)


class DiagonalNormal(ArrayDistribution):
    r"""Multivariate normal distribution with diagonal covariance matrix.

    Implements a multivariate Gaussian distribution with diagonal covariance,
    allowing different means and variances for each dimension. This is simpler
    and more efficient than the full MultivariateNormal when off-diagonal
    correlations are not needed.

    The log density is computed as:

    $$
    \log p(\mathbf{x}) = -\frac{1}{2}\sum_{i=1}^d \left(
    \frac{(x_i - \mu_i)^2}{\sigma_i^2} + \log(2\pi\sigma_i^2)
    \right)
    $$

    Args:
        mean: Mean vector, shape (dim,) or scalar dim.
        scales: Standard deviation vector, shape (dim,).
        rngs: Optional random number generator state.
        var_cls: Variable class for parameters (default: nnx.Param).

    Example:
        >>> # 2D diagonal normal with different means and variances
        >>> mean = jnp.array([1.0, 2.0])
        >>> scales = jnp.array([0.5, 1.5])
        >>> dist = DiagonalNormal(mean, scales)
        >>> samples, log_p = dist.sample(batch_shape=(100,), rng=rng)
        >>> assert samples.shape == (100, 2)
        >>> assert log_p.shape == (100,)

    The parameters an also be instantiated given shapes, and mean is sufficient:
        >>> dist = DiagonalNormal((3,), rngs=nnx.Rngs(0))  # 3D multivariate normal
    """

    def __init__(
        self,
        mean: ParamSpec,
        scales: ParamSpec | None = None,
        *,
        rngs=None,
        var_cls=nnx.Param,
        epsilon: float = 1e-10,
    ):
        """Initialize diagonal multivariate normal distribution.

        Args:
            mean: Mean vector specification.
            scales: Standard deviation vector specification.
            rngs: Optional random number generator state.
            var_cls: Variable class for parameters.
        """
        # Wrap parameters first
        self.mean = default_wrap(
            mean, cls=var_cls, rngs=rngs, init_fn=nnx.initializers.normal()
        )
        if scales is None:
            scales = self.mean.shape
        self.scales_bare = default_wrap(
            scales, cls=var_cls, rngs=rngs, init_fn=nnx.initializers.ones
        )

        super().__init__(rngs=rngs, event_shape=(self.mean.size,))
        self.epsilon = epsilon
        assert self.mean.size == self.scales.size

    @classmethod
    def given_dim(
        cls, dim: int, *, rngs=None, var_cls=nnx.Param, epsilon: float = 1e-10
    ):
        """Create diagonal normal with given dimensionality.

        Args:
            dim: Dimensionality of the distribution.
            rngs: Optional random number generator state.
            var_cls: Variable class for parameters.
            epsilon: Small regularization constant to ensure numerical stability.

        Returns:
            DiagonalNormal instance.
        """
        return cls((dim,), rngs=rngs, var_cls=var_cls, epsilon=epsilon)

    @classmethod
    def given_variances(
        cls,
        mean: ParamSpec,
        variances: ParamSpec,
        *,
        rngs=None,
        var_cls=nnx.Param,
        epsilon: float = 1e-10,
    ):
        """Create diagonal normal with given variances.

        Note: If variances is an instace of nnx.Variable, its value is cloned but the
        type is preserved (nnx.Param, Const, etc.).

        Args:
            variances: Variance vector.
            rngs: Optional random number generator state.
            var_cls: Variable class for parameters.
            epsilon: Small regularization constant to ensure numerical stability.

        Returns:
            DiagonalNormal instance.
        """
        if hasattr(variances, "shape"):
            scales = jnp.sqrt(variances)
            if isinstance(variances, nnx.Variable):
                scales = type(variances)(scales)
        else:
            scales = variances  # only shapes given, these are equal
        return DiagonalNormal(mean, scales, rngs=rngs, var_cls=var_cls, epsilon=epsilon)

    @property
    def dim(self):
        """Dimensionality of the distribution."""
        return self.mean.size

    @property
    def variances(self):
        """Variance vector (scales squared)."""
        return self.scales**2

    @property
    def cov(self):
        """Covariance matrix (diagonal)."""
        return autovmap(1)(jnp.diag)(self.variances)

    @property
    def scales(self):
        return jnp.abs(self.scales_bare.value) + self.epsilon

    def log_density(self, x):
        """Compute log probability density at given points.

        Args:
            x: Points at which to evaluate density, shape (..., dim).

        Returns:
            Log density values with batch dimensions matching input.
        """
        diff = x - self.mean
        scaled_diff = diff / self.scales
        log_density = -0.5 * jnp.sum(scaled_diff**2, axis=-1)
        log_density -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        return log_density - jnp.sum(jnp.log(self.scales), axis=-1)

    def sample(self, batch_shape=(), rng=None):
        """Generate samples from the distribution.

        Args:
            batch_shape: Shape of batch dimensions for vectorized sampling.
            rng: Random key for sampling, or None to use internal rngs.

        Returns:
            Tuple of (samples, log_densities) where samples have shape
            ``(*batch_shape, dim)`` and log_densities have shape ``batch_shape``.
        """
        rng = self._get_rng(rng)
        noise = jax.random.normal(rng, (*batch_shape, self.dim))
        samples = self.mean + self.scales * noise
        return samples, self.log_density(samples)


class MixtureStack(Distribution):
    """Mixture of distributions of equal kind.

    See also :class:`bijx.ScanChain` for a related construct except for composing
    bijections instead of mixing distributions.
    """

    def __init__(self, distributions: Distribution, weights: ParamSpec, *, rngs=None):
        super().__init__(rngs=rngs)
        # need some care to be very sure distributions.rngs aren't used
        distributions = nnx.clone(distributions)
        distributions.rngs = None

        dist_graph, rng_state, dist_vars = nnx.split(distributions, nnx.RngState, ...)
        self.dist_graph = dist_graph
        self.dist_vars = nnx.data(dist_vars)

        assert len(rng_state) == 0, "stack must not carry hidden rngs"

        self.dist_count = jax.tree.leaves(dist_vars)[0].shape[0]
        self.weights = default_wrap(weights, init_fn=nnx.initializers.zeros, rngs=rngs)

    @property
    def weights_normalized(self):
        return nnx.softmax(self.weights.value)

    @property
    def weights_log_normalized(self):
        return nnx.log_softmax(self.weights.value)

    @property
    def stack(self):
        return nnx.merge(self.dist_graph, self.dist_vars)

    def _dist(self, idx: int):
        dist_vars = jax.tree.map(lambda v: v[idx], self.dist_vars)
        return nnx.merge(self.dist_graph, dist_vars)

    def get_batch_shape(self, x):
        return self._dist(0).get_batch_shape(x)

    def log_density(self, x):
        lds = nnx.vmap(lambda d: d.log_density(x), out_axes=-1)(self.stack)
        return jax.nn.logsumexp(lds + self.weights_log_normalized, axis=-1)

    @partial(jax.vmap, in_axes=(None, 0))
    def _sample(self, rng):
        rng_choice, rng_sample = jax.random.split(rng)
        component_indices = jax.random.choice(
            rng_choice,
            self.dist_count,
            p=self.weights_normalized,
        )
        return self._dist(component_indices).sample((), rng=rng_sample)[0]

    def sample(self, batch_shape=(), rng=None):
        total = int(np.prod(batch_shape) if batch_shape else 1)
        rng = self._get_rng(rng)
        samples = self._sample(jax.random.split(rng, total))
        samples = jax.tree.map(lambda x: x.reshape(*batch_shape, *x.shape[1:]), samples)
        log_density = self.log_density(samples)
        return samples, log_density


class GaussianMixture(Distribution):
    """Gaussian mixture model.

    This is a convenience wrapper around :class:`MixtureStack`
    of either diagonal or general multivariate normal distributions.
    """

    def __init__(
        self,
        means: ParamSpec,
        covariances: ParamSpec | None = None,
        weights: ParamSpec | None = None,
        *,
        rngs=None,
        var_cls=nnx.Param,
        epsilon: float = 1e-10,
    ):
        super().__init__(rngs=rngs)
        means = default_wrap(means, init_fn=nnx.initializers.normal(), rngs=rngs)
        mean_shape = means.shape

        cov_shape = covariances.shape if hasattr(covariances, "shape") else covariances

        if len(mean_shape) == len(cov_shape):
            stack = DiagonalNormal.given_variances(
                mean=means,
                variances=covariances,
                rngs=rngs,
                var_cls=var_cls,
                epsilon=epsilon,
            )
        else:
            stack = MultivariateNormal.given_cov(
                mean=means,
                cov=covariances,
                rngs=rngs,
                var_cls=var_cls,
                epsilon=epsilon,
            )

        if weights is None:
            weights = (means.shape[0],)

        self.mixture = MixtureStack(stack, weights, rngs=rngs)

    def get_batch_shape(self, x):
        return self.mixture.get_batch_shape(x)

    @property
    def covs(self):
        return self.mixture.stack.cov

    @property
    def means(self):
        return self.mixture.stack.mean

    @property
    def weights(self):
        return self.mixture.weights_normalized

    def log_density(self, x):
        return self.mixture.log_density(x)

    def sample(self, batch_shape=(), rng=None):
        return self.mixture.sample(batch_shape, rng)
