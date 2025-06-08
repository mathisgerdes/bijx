"""
Probability distributions.
"""

from collections.abc import Callable

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import auto_vmap

from .utils import ShapeInfo, default_wrap


class Distribution(nnx.Module):
    def __init__(self, rngs: nnx.Rngs | None = None):
        self.rngs = rngs

    def _get_rng(self, rng: ftp.PRNGKey | None) -> ftp.PRNGKey:
        if rng is None:
            if self.rngs is None:
                raise ValueError("rngs must be provided")
            rng = self.rngs.sample()
        return rng

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        raise NotImplementedError

    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        raise NotImplementedError

    def log_prob(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        raise NotImplementedError


class ArrayPrior(Distribution):

    def __init__(self, event_shape: tuple[int, ...], rngs: nnx.Rngs | None = None):
        self.event_shape = event_shape
        self.shape_info = ShapeInfo(event_shape=event_shape)
        self.rngs = rngs

    @property
    def event_dim(self):
        return len(self.event_shape)

    @property
    def event_size(self):
        return np.prod(self.event_shape, dtype=int)

    @property
    def event_axes(self):
        return self.shape_info.event_axes

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        return self.shape_info.process_event(x.shape)[0]


class IndependentNormal(ArrayPrior):
    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> jax.Array:
        rng = self._get_rng(rng)
        x = jax.random.normal(rng, batch_shape + self.event_shape)
        return x, self.log_prob(x)

    def log_prob(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.norm.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class IndependentUniform(ArrayPrior):
    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        *,
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> jax.Array:
        rng = self._get_rng(rng)
        x = jax.random.uniform(rng, batch_shape + self.event_shape)
        return x, self.log_prob(x)

    def log_prob(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.uniform.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class DiagonalGMM(Distribution):
    """
    Diagonal Gaussian mixture model.

    Assume data (events) are one-dimensional arrays.
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

    @auto_vmap(x=1)
    def log_prob(self, x):
        """
        Compute the log probability density of the Gaussian mixture model.

        Args:
            x: Input points with shape (batch_size, data_dim)
            nv: Noise variables containing alpha and sigma parameters

        Returns:
            Log probability density for each point in x with shape (batch_size,)
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
        count = np.prod(batch_shape)
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

        log_prob = self.log_prob(samples)

        return samples.reshape(*batch_shape, -1), log_prob.reshape(batch_shape)
