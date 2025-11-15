"""
Composite samplers combining distributions and bijections.

This module provides distribution classes that compose base distributions
with bijections to create more complex sampling mechanisms. These are
fundamental building blocks for normalizing flows and other generative models.

Key classes:

- Transformed: Applies bijection to transform samples from a base distribution
- BufferedSampler: Caches samples to avoid expensive recomputation

All samplers implement the standard :class:`Distribution` interface with
sample() and log_density() methods.
"""

import flax.typing as ftp
import jax
import jax.numpy as jnp
from flax import nnx

from .bijections.base import Bijection
from .distributions import Distribution

__all__ = [
    "Transformed",
    "BufferedSampler",
]


class Transformed(Distribution):
    r"""Distribution obtained by applying a bijection to a base distribution.

    Implements the pushforward distribution
    $p_Y(y) = p_X(f^{-1}(y)) \abs{\det J_{f^{-1}}(y)}$
    where $Y = f(X)$ and $f$ is the bijection.

    The transformed distribution supports both sampling (by transforming samples
    from the base distribution) and density evaluation (by inverse transforming
    and applying the change of variables formula).

    Important: Assumes that the bijection does not change the shape/pytree structure
    of the input in the `log_density` method. This is because it defaults to using
    the given prior to determine the batch shape of the input. If this is not true,
    this class can be extended with a manual implementation of `get_batch_shape`.

    Example:
        >>> prior = bijx.IndependentNormal(event_shape=(2,))
        >>> bijection = bijx.Sigmoid()
        >>> transformed = bijx.Transformed(prior, bijection)
        >>> samples, log_density = transformed.sample((100,), rng=key)

    Args:
        prior: Base distribution to transform.
        bijection: Bijection to apply to samples from the prior.
    """

    def __init__(self, prior: Distribution, bijection: Bijection):
        super().__init__(prior.rngs)
        self.prior = prior
        self.bijection = bijection

    def sample(
        self,
        batch_shape: tuple[int, ...] = (),
        rng: ftp.PRNGKey | None = None,
        **kwargs,
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        """Sample from the transformed distribution.

        Generates samples by first sampling from the base distribution,
        then applying the forward bijection transformation.

        Args:
            batch_shape: Shape of batch dimensions for samples.
            rng: Random key for sampling.
            **kwargs: Additional arguments passed to bijection.

        Returns:
            Tuple of (samples, log_density) where samples have been
            transformed and log_density includes Jacobian correction.
        """
        x, log_density = self.prior.sample(batch_shape, rng=rng, **kwargs)
        x, log_density = self.bijection.forward(x, log_density, **kwargs)
        return x, log_density

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        """Extract batch dimensions from a sample.

        This method defaults to using the prior's `get_batch_shape`
        method to determine the batch shape.

        Args:
            x: A transformed sample.

        Returns:
            Tuple representing the batch shape of the sample.
        """
        return self.prior.get_batch_shape(x)

    def log_density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        """Evaluate log density of the transformed distribution.

        Applies the change of variables formula by inverse transforming
        the input and computing the base distribution density with
        Jacobian correction.

        Args:
            x: Points at which to evaluate log density.
            **kwargs: Additional arguments passed to bijection.

        Returns:
            Log density values at the input points.
        """
        log_density = jnp.zeros(self.get_batch_shape(x))
        x, delta = self.bijection.reverse(x, log_density)
        return self.prior.log_density(x, **kwargs) - delta


class BufferedSampler(Distribution):
    """Distribution wrapper that caches samples for efficient use in MCMC.

    Maintains an internal buffer of pre-computed samples to avoid inefficient
    generation of individual samples.

    The buffer is refilled automatically when exhausted. Only single-sample
    requests (batch_shape=()) use the buffer; batch requests are forwarded
    directly to the underlying distribution.

    Note:
        This class maintains internal state and should be used carefully
        in JAX transformations. The buffer state will be updated during
        sampling operations. This is compatible with flax.nnx's state
        management.

    Example:
        >>> expensive_dist = SomeExpensiveDistribution()
        >>> buffered = bijx.BufferedSampler(expensive_dist, buffer_size=1000)
        >>> sample1 = buffered.sample()  # Fills buffer if empty
        >>> sample2 = buffered.sample()  # Uses cached sample
    """

    def __init__(self, dist: Distribution, buffer_size: int):
        """Initialize buffered sampler.

        Args:
            dist: Base distribution to sample from.
            buffer_size: Number of samples to cache in buffer.
        """
        super().__init__(dist.rngs)
        self.dist = dist
        self.buffer_size = buffer_size

        shapes = nnx.eval_shape(lambda s: s.sample((buffer_size,)), dist)
        self.buffer = nnx.Variable(
            jax.tree.map(lambda s: jnp.empty(s.shape, s.dtype), shapes)
        )
        self.buffer_index = nnx.Variable(jnp.array(buffer_size, dtype=int))

    def sample(
        self, batch_shape: tuple[int, ...] = (), rng: nnx.RngKey | None = None, **kwargs
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        """Sample from the buffered distribution.

        For single samples (batch_shape=()), returns cached samples from the
        buffer, refilling when necessary. For batch requests, forwards directly
        to the underlying distribution.

        Args:
            batch_shape: Shape of batch dimensions. Only () uses buffering.
            rng: Random key for sampling (used when refilling buffer).
            **kwargs: Additional arguments passed to underlying distribution.

        Returns:
            Tuple of (sample, log_density) from buffer or underlying distribution.

        Note:
            The buffer is refilled when exhausted, which updates internal state.
            This may cause issues with JAX transformations that expect pure functions.
        """
        if batch_shape != ():
            return self.dist.sample(batch_shape, rng=rng, **kwargs)

        _, self.buffer_index.value, self.buffer.value = nnx.cond(
            self.buffer_index.value >= self.buffer_size,
            lambda sampler: (
                sampler,
                jnp.zeros_like(self.buffer_index.value),
                sampler.sample(
                    (self.buffer_size,),
                    rng=rng,
                    **kwargs,
                ),
            ),
            lambda sampler: (sampler, self.buffer_index.value + 1, self.buffer.value),
            self.dist,
        )

        draw = jax.tree.map(lambda x: x[self.buffer_index.value], self.buffer.value)
        self.buffer_index.value += 1

        return draw  # (x, log_density)

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        return self.dist.get_batch_shape(x)

    def log_density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        """Evaluate log density using the underlying distribution.

        Args:
            x: Points at which to evaluate log density.
            **kwargs: Additional arguments passed to underlying distribution.

        Returns:
            Log density values from the underlying distribution.
        """
        return self.dist.log_density(x, **kwargs)
