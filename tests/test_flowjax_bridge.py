"""
Tests for FlowJAX bridge compatibility.
"""

import pytest

pytest.importorskip("flowjax")

import flowjax
import flowjax.bijections
import flowjax.distributions
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax import nnx
from flowjax.flows import block_neural_autoregressive_flow

import lfx
import lfx.flowjax as flowjax_bridge


class TestFlowjaxToLfx:
    """Test FlowJAX -> LFX wrapping."""

    def test_simple_affine_bijection(self):
        """Test basic affine bijection wrapping."""
        flowjax_bij = flowjax.bijections.Affine(loc=1.0, scale=2.0)
        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(flowjax_bij)

        x = jnp.array(5.0)
        log_density = jnp.array(0.0)

        y, ld_fwd = lfx_bij.forward(x, log_density)
        x_rec, ld_rev = lfx_bij.reverse(y, log_density)

        np.testing.assert_allclose(x, x_rec, rtol=1e-6)
        np.testing.assert_allclose(ld_fwd + ld_rev, log_density, rtol=1e-6)

    def test_batch_processing(self):
        """Test batch processing with different shapes."""
        flowjax_bij = flowjax.bijections.Exp(shape=(2,))
        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(flowjax_bij)

        x_batch = jr.normal(jr.key(0), (10, 2))
        log_density_batch = jnp.zeros(10)

        y_batch, ld_fwd_batch = lfx_bij.forward(x_batch, log_density_batch)
        x_rec_batch, ld_rev_batch = lfx_bij.reverse(y_batch, log_density_batch)

        np.testing.assert_allclose(x_batch, x_rec_batch, rtol=1e-5)
        np.testing.assert_allclose(
            ld_fwd_batch + ld_rev_batch, log_density_batch, atol=1e-6
        )

    def test_distribution_wrapping(self):
        """Test FlowJAX distribution wrapping."""
        rngs = nnx.Rngs(0)

        key = jr.key(0)
        flow = block_neural_autoregressive_flow(
            key=key,
            base_dist=flowjax.distributions.Normal(jnp.zeros(2)),
        )

        lfx_dist = flowjax_bridge.FlowjaxToLfxDistribution(flow, rngs=rngs)

        samples, log_prob = lfx_dist.sample(batch_shape=(5,))
        assert samples.shape == (5, 2)
        assert log_prob.shape == (5,)

        x = jr.normal(jr.key(1), (2,))
        log_density = lfx_dist.log_density(x)
        assert log_density.shape == ()


class TestLfxToFlowjax:
    """Test LFX -> FlowJAX wrapping."""

    def test_simple_scaling_bijection(self):
        """Test basic scaling bijection wrapping."""
        rngs = nnx.Rngs(0)

        lfx_bij = lfx.bijections.Scaling(jnp.array([2.0, 3.0]), rngs=rngs)

        flowjax_bij = flowjax_bridge.LfxToFlowjaxBijection.from_bijection(
            lfx_bij, shape=(2,)
        )

        x = jnp.array([1.0, 2.0])

        y, log_det_fwd = flowjax_bij.transform_and_log_det(x)
        x_rec, log_det_rev = flowjax_bij.inverse_and_log_det(y)

        np.testing.assert_allclose(x, x_rec, rtol=1e-6)
        np.testing.assert_allclose(log_det_fwd + log_det_rev, 0.0, rtol=1e-6)

    def test_chain_bijection(self):
        """Test chained LFX bijections."""
        rngs = nnx.Rngs(1)

        bij1 = lfx.bijections.Scaling(jnp.array([2.0]), rngs=rngs)
        bij2 = lfx.bijections.Shift(jnp.array([1.0]), rngs=rngs)
        chain = lfx.bijections.Chain(bij1, bij2)

        flowjax_bij = flowjax_bridge.LfxToFlowjaxBijection.from_bijection(
            chain, shape=(1,)
        )

        x = jnp.array([3.0])

        y, log_det_fwd = flowjax_bij.transform_and_log_det(x)
        x_rec, log_det_rev = flowjax_bij.inverse_and_log_det(y)

        np.testing.assert_allclose(x, x_rec, rtol=1e-6)
        np.testing.assert_allclose(log_det_fwd + log_det_rev, 0.0, rtol=1e-6)


class TestHelperFunctions:
    """Test helper conversion functions."""

    def test_to_flowjax_bijection(self):
        """Test to_flowjax helper with bijection."""
        rngs = nnx.Rngs(0)
        lfx_bij = lfx.bijections.Scaling(jnp.array([2.0]), rngs=rngs)

        # Should fail without shape parameter
        with pytest.raises(TypeError):
            flowjax_bridge.to_flowjax(lfx_bij)

        # Should work with shape parameter
        flowjax_bij = flowjax_bridge.to_flowjax(lfx_bij, shape=(1,))
        assert isinstance(flowjax_bij, flowjax_bridge.LfxToFlowjaxBijection)

    def test_from_flowjax_bijection(self):
        """Test from_flowjax helper with bijection."""
        flowjax_bij = flowjax.bijections.Exp()

        lfx_bij = flowjax_bridge.from_flowjax(flowjax_bij)
        assert isinstance(lfx_bij, flowjax_bridge.FlowjaxToLfxBijection)

    def test_from_flowjax_distribution(self):
        """Test from_flowjax helper with distribution."""
        flowjax_dist = flowjax.distributions.Normal(jnp.zeros(2))

        lfx_dist = flowjax_bridge.from_flowjax(flowjax_dist)
        assert isinstance(lfx_dist, flowjax_bridge.FlowjaxToLfxDistribution)

    def test_invalid_module_type(self):
        """Test error handling for invalid module types."""
        with pytest.raises(ValueError, match="Unsupported module type: <class 'str'>"):
            flowjax_bridge.from_flowjax("not a bijection or distribution")

        with pytest.raises(ValueError, match="Unsupported module type: <class 'str'>"):
            flowjax_bridge.to_flowjax("not a bijection or distribution")


class TestRoundTripConsistency:
    """Test round-trip consistency: LFX -> FlowJAX -> LFX and vice versa."""

    def test_lfx_flowjax_lfx_bijection(self):
        """Test LFX -> FlowJAX -> LFX bijection round trip."""
        rngs = nnx.Rngs(42)

        original_lfx = lfx.bijections.Scaling(jnp.array([1.5, 2.0]), rngs=rngs)

        flowjax_bij = flowjax_bridge.LfxToFlowjaxBijection.from_bijection(
            original_lfx, shape=(2,)
        )

        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(flowjax_bij)

        x = jnp.array([3.0, 4.0])
        log_density = jnp.array(0.0)

        y1, ld1 = original_lfx.forward(x, log_density)
        y2, ld2 = lfx_bij.forward(x, log_density)

        np.testing.assert_allclose(y1, y2, rtol=1e-6)
        np.testing.assert_allclose(ld1, ld2, rtol=1e-6)

    def test_flowjax_lfx_flowjax_bijection(self):
        """Test FlowJAX -> LFX -> FlowJAX bijection round trip."""
        original_flowjax = flowjax.bijections.Affine(
            loc=jnp.array([1.0, 2.0]), scale=jnp.array([2.0, 3.0])
        )

        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(original_flowjax)

        flowjax_bij = flowjax_bridge.LfxToFlowjaxBijection.from_bijection(
            lfx_bij, shape=(2,)
        )

        x = jnp.array([3.0, 4.0])

        y1, ld1 = original_flowjax.transform_and_log_det(x)
        y2, ld2 = flowjax_bij.transform_and_log_det(x)

        np.testing.assert_allclose(y1, y2, rtol=1e-6)
        np.testing.assert_allclose(ld1, ld2, rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_scalar_bijections(self):
        """Test scalar bijections."""
        flowjax_bij = flowjax.bijections.Tanh()
        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(flowjax_bij)

        x = jnp.array(1.0)
        log_density = jnp.array(0.0)

        y, ld_fwd = lfx_bij.forward(x, log_density)
        x_rec, ld_rev = lfx_bij.reverse(y, log_density)

        np.testing.assert_allclose(x, x_rec, rtol=1e-6)
        np.testing.assert_allclose(ld_fwd + ld_rev, log_density, rtol=1e-6)

    def test_zero_log_density(self):
        """Test that log density handling is correct."""
        flowjax_bij = flowjax.bijections.Exp()
        lfx_bij = flowjax_bridge.FlowjaxToLfxBijection(flowjax_bij)

        x = jnp.array(1.0)
        log_density = jnp.array(0.0)

        y_lfx, ld_lfx = lfx_bij.forward(x, log_density)
        y_fj, log_det_fj = flowjax_bij.transform_and_log_det(x)

        np.testing.assert_allclose(y_lfx, y_fj, rtol=1e-6)
        np.testing.assert_allclose(ld_lfx, log_density - log_det_fj, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
