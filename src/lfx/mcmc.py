"""
Markov Chain Monte Carlo methods.
"""

import typing as tp

import flax
import jax
import jax.numpy as jnp
from flax import nnx


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

    def __init__(self, sampler, target_log_prob: tp.Callable):
        self.sampler = sampler
        self.target_log_prob = target_log_prob

    def propose(self, rng):
        position, log_prob_proposal = self.sampler.sample(rng=rng)
        log_prob_target = self.target_log_prob(position)
        return IMHState(position, log_prob_target, log_prob_proposal)

    def init(self, rng):
        return self.propose(rng)

    def step(self, rng, state):
        rng_proposal, rng_uniform = jax.random.split(rng)
        proposal = self.propose(rng_proposal)

        log_alpha = (
            proposal.log_prob_target
            - proposal.log_prob_proposal
            - state.log_prob_target
            + state.log_prob_proposal
        )
        log_uniform = jnp.log(jax.random.uniform(rng_uniform))
        is_accepted = log_uniform < log_alpha

        accept_prob = jnp.minimum(1.0, jnp.exp(log_alpha))

        new_state = nnx.cond(
            is_accepted,
            lambda: proposal,
            lambda: state,
        )

        info = IMHInfo(is_accepted, accept_prob, proposal)

        return new_state, info
