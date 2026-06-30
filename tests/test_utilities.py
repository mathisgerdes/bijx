"""
Utilities & Bridges — General utilities tests.

Covers: `default_wrap`, `ShapeInfo`, `effective_sample_size`, `reverse_dkl`,
`moving_average`, `noise_model`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bijx import (
    Const,
    IndependentNormal,
    ShapeInfo,
    default_wrap,
    effective_sample_size,
    moving_average,
    noise_model,
    reverse_dkl,
)
from bijx.mcmc import IMH, IMHInfo, IMHState

from .utils import assert_finite_and_real


class TestDefaultWrap:
    def test_wrap_array(self):
        val = jnp.array([1.0, 2.0])
        wrapped = default_wrap(val)
        assert isinstance(wrapped, nnx.Param)
        np.testing.assert_allclose(wrapped.get_value(), val)

    def test_wrap_shape(self, rng_key):
        wrapped = default_wrap((3, 2), rngs=nnx.Rngs(rng_key))
        assert isinstance(wrapped, nnx.Param)
        assert wrapped.shape == (3, 2)

    def test_wrap_variable_passthrough(self):
        v = nnx.Param(jnp.array([3.0]))
        assert default_wrap(v) is v


class TestShapeInfo:
    def test_infer_shapes(self):
        info = ShapeInfo(event_shape=(8, 8, 3), space_dim=2)
        assert info.space_shape == (8, 8)
        assert info.channel_shape == (3,)
        assert info.event_axes == (-3, -2, -1)
        assert info.space_axes == (-3, -2)
        assert info.channel_axes == (-1,)

    def test_process_and_flatten(self):
        info = ShapeInfo(space_dim=2, channel_dim=0)
        x = jnp.zeros((4, 5, 6))
        flat, _, sub = info.process_and_flatten(x)
        assert flat.shape == (4, 30)
        assert sub.space_dim == 2
        assert sub.channel_dim == 0

    def test_process_and_canonicalize(self):
        info = ShapeInfo(space_dim=1, channel_dim=1)
        x = jnp.zeros((2, 3, 4, 5))
        canonical, batch_shape, sub = info.process_and_canonicalize(x)
        assert canonical.shape == (
            6,
            4,
            5,
        )  # batch_size=6, space_shape=(4,), channel_size=5
        assert batch_shape == (2, 3)
        assert sub.space_dim == 1
        assert sub.channel_dim == 1


class TestESSandKL:
    def test_effective_sample_size_bounds(self, rng_key):
        key = rng_key
        # Simulate two log-densities
        target = jax.random.normal(key, (100,))
        sample = target + 0.1 * jax.random.normal(jax.random.split(key)[0], (100,))
        ess = effective_sample_size(target, sample)
        assert jnp.isscalar(ess)
        assert 0.0 <= ess <= 1.0

    def test_reverse_dkl_simple(self):
        # Constant shift; KL should equal shift on average
        x = jnp.array([0.0, 1.0, 2.0])
        target_ld = -x  # log p
        sample_ld = -(x + 1.0)  # log q = log p - 1
        dkl = reverse_dkl(target_ld, sample_ld)
        np.testing.assert_allclose(dkl, -1.0)


class TestMovingAverage:
    def test_window_basic(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = moving_average(x, window=3)
        np.testing.assert_allclose(y, jnp.array([2.0, 3.0, 4.0]))

    def test_short_series(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = moving_average(x, window=3)
        np.testing.assert_allclose(y, jnp.array([2.0]))


class TestNoiseModel:
    class _Toy(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.ones((3,)))
            self.frozen = Const(0.0)

    def test_noise_added(self, rng_key):
        model = self._Toy()
        noisy = noise_model(nnx.Rngs(rng_key), model, scale=0.5)
        assert not jnp.allclose(noisy.w.get_value(), model.w.get_value())

    def test_filtering(self, rng_key):
        model = self._Toy()
        # Explicitly target trainable parameters only
        noisy = noise_model(nnx.Rngs(rng_key), model, 1.0, nnx.Param)
        # Param 'w' should change; 'frozen' should remain constant
        assert not jnp.allclose(noisy.w, model.w)
        assert noisy.frozen == 0.0


class TestMCMCSamplers:
    """Basic tests for MCMC functionality."""

    def test_imh_proposal_log_probs_consistent(self, rng_key):
        """propose() stores log-probs consistent with the target and proposal."""

        def target_log_prob(x):
            return -0.5 * jnp.sum(x**2)

        proposal = IndependentNormal(event_shape=(2,))
        sampler = IMH(proposal, target_log_prob)

        rngs = nnx.Rngs(rng_key)
        prop_state = sampler.propose(rngs())

        # Stored log-probs must match a direct evaluation at the sampled point.
        np.testing.assert_allclose(
            prop_state.log_prob_target, target_log_prob(prop_state.position)
        )
        np.testing.assert_allclose(
            prop_state.log_prob_proposal, proposal.log_density(prop_state.position)
        )

    def test_imh_step_returns_proposal_or_current(self, rng_key):
        """A step lands on either the proposal or the unchanged current state."""

        def target_log_prob(x):
            return -0.5 * jnp.sum(x**2)

        proposal = IndependentNormal(event_shape=(2,))
        sampler = IMH(proposal, target_log_prob)

        rngs = nnx.Rngs(rng_key)
        state = sampler.init(rngs())
        new_state, info = sampler.step(rngs(), state)

        assert isinstance(new_state, IMHState)
        assert isinstance(info, IMHInfo)
        # The new position is exactly the proposed one (accepted) or the old one.
        if bool(info.is_accepted):
            np.testing.assert_array_equal(new_state.position, info.proposal.position)
        else:
            np.testing.assert_array_equal(new_state.position, state.position)

    def test_imh_perfect_proposal_always_accepts(self, rng_key):
        """When the proposal equals the target, every IMH move is accepted.

        With ``target_log_prob == proposal.log_density`` the Metropolis ratio is
        identically 1, so acceptance is exact regardless of the draw — a strong
        correctness check on the acceptance formula.
        """
        proposal = IndependentNormal(event_shape=(2,))
        sampler = IMH(proposal, proposal.log_density)

        rngs = nnx.Rngs(rng_key)
        state = sampler.init(rngs())

        for _ in range(20):
            state, info = sampler.step(rngs(), state)
            assert bool(info.is_accepted)
            np.testing.assert_allclose(info.accept_prob, 1.0)

    def test_sampling_mcmc_consistency(self, rng_key):
        """Test that the MCMC chain concentrates around the target mean.

        The chain is run with a single ``lax.scan`` (one compile, no Python-side
        per-step dispatch) instead of a long Python loop.
        """
        # Simple 1D target: N(1, 0.5^2)
        target_mean, target_std = 1.0, 0.5

        def target_log_prob(x):
            return -0.5 * ((x - target_mean) / target_std) ** 2

        # Proposal: N(0, 1)
        proposal = IndependentNormal(event_shape=())
        sampler = IMH(proposal, target_log_prob)

        rngs = nnx.Rngs(rng_key)
        state = sampler.init(rngs())

        n_steps = 200
        keys = jax.random.split(rngs(), n_steps)

        def body(carry_state, key):
            new_state, _ = sampler.step(key, carry_state)
            return new_state, new_state.position

        _, positions = jax.lax.scan(body, state, keys)

        # Discard the first half as burn-in
        samples = positions[n_steps // 2 :]
        assert_finite_and_real(samples, "MCMC samples")

        # Mean should be roughly in the right direction
        mean_diff = jnp.mean(samples) - target_mean
        assert abs(mean_diff) < 0.2  # Very loose bound
