import typing as tp

import flax
import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .core.base import Bijection
from .utils import ShapeInfo


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


class Sampler(Distribution):

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

    def log_prob(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        log_density = jnp.zeros(self.prior.get_batch_shape(x))
        x, delta = self.bijection.reverse(x, log_density)
        return self.prior.log_prob(x, **kwargs) - delta


class BufferedSampler(Sampler):
    """Buffers samples from a sampler to avoid recomputing them."""

    def __init__(self, sampler: Sampler, buffer_size: int):
        super().__init__(sampler.rngs)
        self.sampler = sampler
        self.buffer_size = buffer_size

        shapes = nnx.eval_shape(lambda s: s.sample((buffer_size,)), sampler)
        self.buffer = nnx.Variable(
            jax.tree.map(lambda s: jnp.empty(s.shape, s.dtype), shapes)
        )
        self.buffer_index = nnx.Variable(jnp.array(buffer_size, dtype=int))

    def sample(
        self, batch_shape: tuple[int, ...] = (), rng: nnx.RngKey | None = None, **kwargs
    ) -> tuple[ftp.ArrayPytree, jax.Array]:
        if batch_shape != ():
            return self.sampler.sample(batch_shape, rng=rng, **kwargs)

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
            self.sampler,
        )

        sample = jax.tree.map(lambda x: x[self.buffer_index.value], self.buffer.value)
        self.buffer_index.value += 1

        return sample

    def log_prob(self, x: ftp.ArrayPytree, **kwargs) -> jax.Array:
        return self.sampler.log_prob(x, **kwargs)


# ------------------------------------------------------------------------------
# Metropolis-Hastings
# ------------------------------------------------------------------------------


# independent Metropolis-Hastings


@flax.struct.dataclass
class IMHState:
    position: flax.typing.ArrayPytree
    log_prob_target: float
    log_prob_proposal: float


@flax.struct.dataclass
class IMHInfo:
    is_accepted: bool
    accept_prob: float
    proposal: IMHState


class IMH(nnx.Module):
    """
    Independent Metropolis-Hastings

    Roughly modeled after blackjax API, but note that the sampler is
    expected to return "position" and proposal log-probabilities.

    Note: difference to blackjax.irmh is that we produce log-likelihoods
    at same time as samples
    """

    def __init__(self, sampler: Sampler, target_log_prob: tp.Callable):
        self.sampler = sampler
        self.target_log_prob = target_log_prob

    def propose(self, rng):
        position, log_prob_proposal = self.sampler.sample(rng=rng)
        return IMHState(
            position=position,
            log_prob_proposal=log_prob_proposal,
            log_prob_target=self.target_log_prob(position),
        )

    def init(self, rng):
        return self.propose(rng)

    def step(self, rng, state):
        proposal = self.propose(rng)

        accept_prob = jnp.exp(
            proposal.log_prob_target
            - proposal.log_prob_proposal
            - state.log_prob_target
            + state.log_prob_proposal
        )
        accept_prob = jnp.minimum(accept_prob, 1.0)
        is_accepted = jax.random.bernoulli(rng, accept_prob)
        new_state = jax.lax.cond(
            is_accepted,
            lambda prev, prop: prop,
            lambda prev, prop: prev,
            state,
            proposal,
        )
        return new_state, IMHInfo(
            is_accepted=is_accepted,
            accept_prob=accept_prob,
            proposal=proposal,
        )
