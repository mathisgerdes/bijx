"""
Samplers that compose distributions and bijections.
"""

import flax.typing as ftp
import jax
import jax.numpy as jnp
from flax import nnx

from .bijections.base import Bijection
from .distributions import Distribution


class Transformed(Distribution):

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
        x, log_density = self.prior.sample(batch_shape, rng=rng, **kwargs)
        x, log_density = self.bijection.forward(x, log_density, **kwargs)
        return x, log_density

    def log_density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        log_density = jnp.zeros(self.prior.get_batch_shape(x))
        x, delta = self.bijection.reverse(x, log_density)
        return self.prior.log_density(x, **kwargs) - delta


class BufferedSampler(Distribution):
    """Buffers samples from a sampler to avoid recomputing them."""

    def __init__(self, dist: Distribution, buffer_size: int):
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

        sample = jax.tree.map(lambda x: x[self.buffer_index.value], self.buffer.value)
        self.buffer_index.value += 1

        return sample

    def log_density(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        return self.dist.log_density(x, **kwargs)
