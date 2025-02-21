from functools import partial

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .bijections import Bijection


class Prior(nnx.Module):
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


class ArrayPrior(Prior):

    def __init__(self, event_shape: tuple[int, ...], rngs: nnx.Rngs | None = None):
        self.event_shape = event_shape
        self.rngs = rngs

    @property
    def event_dim(self):
        return len(self.event_shape)

    @property
    def event_size(self):
        return np.prod(self.event_shape, dtype=int)

    @property
    def event_axes(self):
        return tuple(range(-1, -self.event_dim - 1, -1))

    def get_batch_shape(self, x: ftp.ArrayPytree) -> tuple[int, ...]:
        return x.shape[:-self.event_dim]


class IndependentNormal(ArrayPrior):
    def sample(self, batch_shape: tuple[int, ...] = (), *, rng: ftp.PRNGKey | None = None, **kwargs) -> jax.Array:
        rng = self._get_rng(rng)
        x = jax.random.normal(rng, batch_shape + self.event_shape)
        return x, self.log_prob(x)

    def log_prob(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.norm.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class IndependentUniform(ArrayPrior):
    def sample(self, batch_shape: tuple[int, ...] = (), *, rng: ftp.PRNGKey | None = None, **kwargs) -> jax.Array:
        rng = self._get_rng(rng)
        x = jax.random.uniform(rng, batch_shape + self.event_shape)
        return x, self.log_prob(x)

    def log_prob(self, x: ftp.Array, **kwargs) -> jax.Array:
        logp = jax.scipy.stats.uniform.logpdf(x)
        logp = jnp.sum(logp, axis=self.event_axes)
        return logp


class Sampler(Prior):

    def __init__(self, prior: Prior, bijection: Bijection):
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
        x, log_density = self.bijection(x, log_density, **kwargs)
        return x, log_density

    def log_prob(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        log_density = jnp.zeros(self.prior.get_batch_shape(x))
        x, delta = self.bijection.reverse(x, log_density)
        return self.prior.log_prob(x, **kwargs) - delta


class BufferedSampler(Sampler):
    """Buffers samples from a sampler to avoid recomputing them."""

    def __init__(self, sampler: Sampler, buffer_size: int):
        self.sampler = sampler
        self.buffer_size = buffer_size

        shapes = nnx.eval_shape(lambda s: s.sample((buffer_size,)), sampler)
        self.buffer = nnx.Variable(
            jax.tree.map(lambda s: jnp.empty(s.shape, s.dtype), shapes)
        )
        self.buffer_index = nnx.Variable(buffer_size)

    def sample(
        self,
        rng: nnx.RngKey | None = None,
        **kwargs
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        _, self.buffer_index.value, self.buffer.value = nnx.cond(
            self.buffer_index.value >= self.buffer_size,
            lambda sampler: (sampler, 0, sampler.sample(
                (self.buffer_size,),
                rng=rng,
                **kwargs,
            )),
            lambda sampler: (sampler, self.buffer_index.value + 1, self.buffer.value),
            self.sampler,
        )

        sample = jax.tree.map(
            lambda x: x[self.buffer_index.value],
            self.buffer.value
        )
        self.buffer_index.value += 1

        return sample

    def log_prob(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        return self.sampler.log_prob(x, **kwargs)
