"""
Markov Chain Monte Carlo sampling algorithms.

This module provides MCMC algorithms specifically in the context where samples
are generated independently from normalizing flows.
The implementations are designed to integrate with the bijx
ecosystem and follow similar API patterns to blackjax.
"""

import typing as tp
from dataclasses import dataclass, replace

import flax
import jax
import jax.numpy as jnp
from flax import nnx

__all__ = ["IMH", "IMHState", "IMHInfo"]


@dataclass(frozen=True)
class IMHState(nnx.Pytree):
    """State for Independent Metropolis-Hastings sampler.

    Stores current position and associated log probabilities for both
    target and proposal distributions.
    """

    position: flax.typing.ArrayPytree
    """Current sample position in the state space."""
    log_prob_target: float
    """Log probability of position under target distribution."""
    log_prob_proposal: float
    """Log probability of position under proposal distribution."""


@dataclass(frozen=True)
class IMHInfo(nnx.Pytree):
    """Information about the IMH sampling step."""

    is_accepted: bool
    """Whether the proposed move was accepted."""
    accept_prob: float
    """Acceptance probability for the proposed move."""
    proposal: IMHState
    """The proposed state that was considered."""

    def replace(self, **changes):
        """Create new config with specified parameters replaced."""
        return replace(self, **changes)


class IMH(nnx.Module):
    r"""Independent Metropolis-Hastings sampler.

    Implements the Independent Metropolis-Hastings algorithm for sampling from
    a target distribution using an independent proposal distribution. This is
    particularly useful when the proposal distribution is a good approximation
    to the target, such as using a normalizing flow as the proposal.

    The algorithm generates proposals independently from the current state,
    then accepts or rejects them based on the Metropolis criterion:

    $$\alpha = \min\left(1, \frac{p(x')q(x)}{p(x)q(x')}\right)$$

    where $p(x)$ is the target density and $q(x)$ is the proposal density.

    Key differences from blackjax.irmh:
        - Sampler returns both samples and their log probabilities
        - Integration with bijx distribution/bijection ecosystem
        - Flax NNX module system compatibility

    Example:
        >>> target_log_prob = lambda x: -0.5 * jnp.sum(x**2)  # Standard normal
        >>> proposal = bijx.IndependentNormal(event_shape=(2,))
        >>> sampler = bijx.IMH(proposal, target_log_prob)
        >>> initial_state = sampler.init(key)
        >>> new_state, info = sampler.step(key, initial_state)
    """

    def __init__(self, sampler, target_log_prob: tp.Callable):
        """Initialize Independent Metropolis-Hastings sampler.

        Args:
            sampler: Proposal distribution implementing sample() method that
                returns (position, log_prob) tuples.
            target_log_prob: Function computing log probability density of
                target distribution.
        """
        self.sampler = sampler
        self.target_log_prob = target_log_prob

    def propose(self, rng):
        """Generate a proposal state.

        Samples from the proposal distribution and evaluates both proposal
        and target log probabilities at the sampled position.

        Args:
            rng: Random key for sampling.

        Returns:
            IMHState containing the proposal position and log probabilities.
        """
        position, log_prob_proposal = self.sampler.sample(rng=rng)
        log_prob_target = self.target_log_prob(position)
        return IMHState(position, log_prob_target, log_prob_proposal)

    def init(self, rng):
        """Initialize the sampler state.

        Args:
            rng: Random key for initialization.

        Returns:
            Initial IMHState by proposing from the proposal distribution.
        """
        return self.propose(rng)

    def step(self, rng, state):
        r"""Perform one step of Independent Metropolis-Hastings.

        Generates a proposal and applies the Metropolis acceptance criterion
        to decide whether to move to the new state or remain at the current one.

        Args:
            rng: Random key for the step.
            state: Current IMHState of the chain.

        Returns:
            Tuple of (new_state, info) where:
            - new_state: Updated IMHState (either proposal or unchanged)
            - info: IMHInfo with acceptance decision and diagnostics

        Note:
            The acceptance probability is computed as:
            $\alpha = \min(1, \exp(\log p(x') - \log q(x') - \log p(x) + \log q(x)))$
        """
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
