"""
Tests for FlowJAX bridge in bijx.flowjax.

Lightweight proof-of-principle checks that the adapters work and gradients
can be computed through both directions on a simple loss.
"""

import pytest

pytest.importorskip("flowjax")

import flowjax
import flowjax.bijections
import flowjax.distributions
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax import nnx

import bijx
import bijx.flowjax as fj_bridge

# Central test tolerances
from .utils import ATOL, RTOL, RTOL_RELAXED


def _assert_bijection_inverse_consistency(bijection, x, ld, rtol=RTOL):
    """Helper to test bijection inverse consistency."""
    y, ld_forward = bijection.forward(x, ld)
    x_reconstructed, ld_reverse = bijection.reverse(y, ld)
    np.testing.assert_allclose(x_reconstructed, x, rtol=rtol)
    np.testing.assert_allclose(ld_forward + ld_reverse, ld, rtol=rtol)


def _assert_flowjax_inverse_consistency(bijection, x, rtol=RTOL):
    """Helper to test FlowJAX bijection inverse consistency."""
    y, ld_forward = bijection.transform_and_log_det(x)
    x_reconstructed, ld_reverse = bijection.inverse_and_log_det(y)
    np.testing.assert_allclose(x_reconstructed, x, rtol=rtol)
    np.testing.assert_allclose(ld_forward + ld_reverse, 0.0, rtol=rtol)


class TestFlowjaxToBijx:
    def test_affine_inverse_and_log(self):
        fj = flowjax.bijections.Affine(loc=1.0, scale=2.0)
        bx = fj_bridge.FlowjaxToBijxBijection(fj)
        x = jnp.array(5.0)
        ld = jnp.array(0.0)
        _assert_bijection_inverse_consistency(bx, x, ld)

    def test_batch_shape(self, rng_key):
        fj = flowjax.bijections.Exp(shape=(2,))
        bx = fj_bridge.FlowjaxToBijxBijection(fj)
        x = jr.normal(rng_key, (7, 2))
        ld = jnp.zeros((7,))
        y, ld1 = bx.forward(x, ld)
        x_reconstructed, ld2 = bx.reverse(y, ld)
        np.testing.assert_allclose(x_reconstructed, x, rtol=RTOL_RELAXED)
        np.testing.assert_allclose(ld1 + ld2, ld, atol=ATOL)


class TestBijxToFlowjax:
    def test_simple_scaling(self, rng_key):
        bij = bijx.bijections.Scaling(jnp.array([2.0, 3.0]), rngs=nnx.Rngs(rng_key))
        fj = fj_bridge.BijxToFlowjaxBijection.from_bijection(bij, shape=(2,))
        x = jnp.array([1.0, 2.0])
        _assert_flowjax_inverse_consistency(fj, x)

    def test_chain(self, rng_key):
        r = nnx.Rngs(rng_key)
        s = bijx.bijections.Scaling(jnp.array([2.0]), rngs=r)
        t = bijx.bijections.Shift(jnp.array([1.0]), rngs=r)
        chain = bijx.bijections.Chain(s, t)
        fj = fj_bridge.BijxToFlowjaxBijection.from_bijection(chain, shape=(1,))
        x = jnp.array([3.0])
        _assert_flowjax_inverse_consistency(fj, x)


class TestHelpers:
    def test_helper_round_trip(self, rng_key):
        """Test that round-trip conversion preserves behavior."""
        bx = bijx.bijections.Scaling(jnp.array([1.5, 2.0]), rngs=nnx.Rngs(rng_key))
        fj = fj_bridge.to_flowjax(bx, shape=(2,))
        bx2 = fj_bridge.from_flowjax(fj)
        x = jnp.array([3.0, 4.0])
        ld = jnp.array(0.0)
        y1, ld1 = bx.forward(x, ld)
        y2, ld2 = bx2.forward(x, ld)
        np.testing.assert_allclose(y1, y2, rtol=RTOL)
        np.testing.assert_allclose(ld1, ld2, rtol=RTOL)

    def test_error_paths(self, rng_key):
        """Test that invalid usage raises appropriate errors."""
        # Invalid module type
        msg = "Unsupported module type: <class 'str'>"
        with pytest.raises(ValueError, match=msg):
            fj_bridge.from_flowjax("not-a-module")

        # Missing shape parameter
        msg = "Converting bijx bijection to FlowJAX requires 'shape' parameter"
        with pytest.raises(TypeError, match=msg):
            fj_bridge.to_flowjax(bijx.bijections.Identity)

        bij = bijx.bijections.Scaling(jnp.array([2.0]), rngs=nnx.Rngs(rng_key))

        # Conditional misuse: providing condition without cond_shape
        fj = fj_bridge.BijxToFlowjaxBijection.from_bijection(bij, shape=(1,))
        msg = "Condition provided but cond_shape is None"
        with pytest.raises(TypeError, match=msg):
            _ = fj.transform_and_log_det(jnp.array([1.0]), condition=jnp.array([0.0]))

        # Wrong conditional trailing shape
        fjc = fj_bridge.BijxToFlowjaxBijection.from_bijection(
            bij, shape=(1,), cond_shape=(2,)
        )
        msg = "Expected condition.shape \\(2,\\); got \\(1,\\)"
        with pytest.raises(ValueError, match=msg):
            _ = fjc.transform_and_log_det(jnp.array([1.0]), condition=jnp.array([0.0]))


class TestGradients:
    def test_grad_through_bijx_wrapped_in_flowjax(self, rng_key):
        def loss(scale_val):
            # Rebuild bijection with new parameter for grad.
            bx = bijx.bijections.Scaling(jnp.array(scale_val), rngs=nnx.Rngs(rng_key))
            fj = fj_bridge.to_flowjax(bx, shape=())
            y, ld = fj.transform_and_log_det(jnp.array(1.0))
            return y**2 + ld**2

        g = jax.grad(loss)(2.0)
        assert jnp.isfinite(g)

    def test_grad_through_flowjax_wrapped_in_bijx(self):
        def loss(scale_val):
            fj = flowjax.bijections.Affine(loc=1.0, scale=scale_val)
            bx = fj_bridge.from_flowjax(fj)
            y, ld = bx.forward(jnp.array(1.0), jnp.array(0.0))
            return y**2 + ld**2

        g = jax.grad(loss)(2.0)
        assert jnp.isfinite(g)


class TestDistributionBridge:
    def test_flowjax_dist_to_bijx(self, rng_key):
        """Test FlowJAX distribution to bijx distribution conversion."""
        key = rng_key
        flow = flowjax.distributions.Normal(jnp.zeros(2))
        bx = fj_bridge.FlowjaxToBijxDistribution(flow, rngs=nnx.Rngs(rng_key))
        samples, logp = bx.sample(batch_shape=(5,), rng=key)
        assert samples.shape == (5, 2)
        assert logp.shape == (5,)
