"""
Targeted tests for coupling layers and related utilities.

Covers:
- GeneralCouplingLayer for split vs multiplicative masking
- bijection_event_rank branches and error paths
- AutoVmapReconstructor behavior with batched params and input_ranks
- BinaryMask advanced API (extra_channel_dims) and negative paths
"""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bijx import (
    AffineLinear,
    BinaryMask,
    GeneralCouplingLayer,
    ModuleReconstructor,
    checker_mask,
)

from .utils import ATOL, RTOL, assert_finite_and_real


class DummyUnitJacobian(nnx.Module):
    """
    Bijection that leaves data unchanged and adds +1 to the log_density everywhere.
    Contains unused parameter to ensure non-zero parameter size for reconstruction.
    """

    def __init__(self, *, rngs=None):
        # Unused parameter; ensures params_total_size > 0
        self.theta = nnx.Param(jnp.array(0.0))

    # Behaves like a bijection: forward/reverse signatures
    def forward(self, x, log_density):
        return x, log_density + jnp.ones_like(log_density)

    def reverse(self, x, log_density):
        return x, log_density - jnp.ones_like(log_density)


class ZeroNet(nnx.Module):
    """Simple network returning zeros with a specified final feature dimension."""

    def __init__(self, out_features):
        self.out_features = out_features

    def __call__(self, x):
        batch_shape = x.shape[:-1]
        return jnp.zeros(batch_shape + (self.out_features,), dtype=x.dtype)


class TestGeneralCouplingLayer:
    def _build_layer(
        self,
        mask: BinaryMask,
        template_module: nnx.Module,
        *,
        split: bool,
        bij_rank: int,
    ):
        template = ModuleReconstructor(template_module)
        param_count = template.params_total_size
        # network maps passive -> (count_active * param_count)
        # then reshape to (..., count_active, param_count)
        count_active, count_passive = mask.counts
        net = ZeroNet(out_features=count_active * param_count)

        def reshape_params(p):
            if split:
                # params per active position
                return p.reshape(p.shape[:-1] + (count_active, param_count))
            # multiplicative masking: params for full event grid
            event_size = mask.event_size
            # Rebuild net with appropriate output size if needed
            if p.shape[-1] != event_size * param_count:
                p = jnp.zeros(p.shape[:-1] + (event_size * param_count,), dtype=p.dtype)
            return p.reshape(p.shape[:-1] + (event_size, param_count))

        param_net = nnx.Sequential(net, reshape_params)
        return GeneralCouplingLayer(
            param_net, mask, template, bijection_event_rank=bij_rank, split=split
        )

    def test_split_scalar_rank0_identity_affine_template(self, rng_key):
        # Mask with half active entries
        mask = checker_mask((4,), parity=True)
        # Use AffineLinear template;
        # ZeroNet outputs zeros -> scale=exp(0)=1, shift=0 -> identity
        layer = self._build_layer(
            mask, AffineLinear(rngs=nnx.Rngs(rng_key)), split=True, bij_rank=0
        )

        x = jnp.arange(8.0).reshape(2, 4)
        ld = jnp.zeros((2,))
        y, ld_f = layer.forward(x, ld)
        x_b, ld_b = layer.reverse(y, ld_f)

        np.testing.assert_allclose(x_b, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_b, ld, atol=ATOL, rtol=RTOL)
        assert_finite_and_real(y, "split rank0 forward output")
        assert_finite_and_real(ld_f, "split rank0 forward log density")

    def test_split_scalar_rank0_logdensity_counts_active(self):
        # With DummyUnitJacobian, Δlogp should equal number of active entries
        mask = checker_mask((6,), parity=True)
        count_active, _ = mask.counts

        layer = self._build_layer(mask, DummyUnitJacobian(), split=True, bij_rank=0)
        x = jnp.ones((3, 6))
        ld0 = jnp.zeros((3,))
        _, ld1 = layer.forward(x, ld0)
        np.testing.assert_allclose(ld1, jnp.full((3,), count_active), rtol=RTOL)

    def test_multiplicative_mask_rank0_inverse_consistency(self, rng_key):
        # Exercise split=False branch;
        # use identity-affine so numerical behavior is robust
        mask = checker_mask((5,), parity=False)
        layer = self._build_layer(
            mask, AffineLinear(rngs=nnx.Rngs(rng_key)), split=False, bij_rank=0
        )
        x = jnp.linspace(0.0, 1.0, 10).reshape(2, 5)
        ld0 = jnp.zeros((2,))
        y, ld1 = layer.forward(x, ld0)
        xb, ldb = layer.reverse(y, ld1)
        np.testing.assert_allclose(xb, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ldb, ld0, atol=ATOL, rtol=RTOL)

    def test_split_vector_rank1_no_event_sum(self):
        # For bijection_event_rank=1 and split=True,
        # Δlogp should increment by 1, not by count_active
        mask = checker_mask((4,), parity=True)

        layer = self._build_layer(mask, DummyUnitJacobian(), split=True, bij_rank=1)
        x = jnp.ones((2, 4))
        ld0 = jnp.zeros((2,))
        _, ld1 = layer.forward(x, ld0)
        # Current implementation returns per-active contributions without reduction
        # for rank=1.
        assert ld1.shape[-1] == mask.count_primary
        np.testing.assert_allclose(ld1, jnp.ones_like(ld1), rtol=RTOL)

    def test_error_split_rank_gt1_raises(self):
        mask = checker_mask((3,), parity=True)
        layer = self._build_layer(mask, DummyUnitJacobian(), split=True, bij_rank=2)
        msg = "bijection_event_rank must be 0 or 1"
        with pytest.raises(ValueError, match=msg):
            layer.forward(jnp.ones((1, 3)), jnp.zeros((1,)))

    def test_error_multiplicative_rank_exceeds_event_shape(self):
        # mask.event_shape has rank 1; bijection_event_rank=2 should error
        mask = checker_mask((3,), parity=False)
        layer = self._build_layer(mask, DummyUnitJacobian(), split=False, bij_rank=2)
        msg = "Event rank given mask shape \\(3,\\) is too low"
        with pytest.raises(ValueError, match=msg):
            layer.forward(jnp.ones((2, 3)), jnp.zeros((2,)))


class TestBinaryMaskAdvanced:
    def test_indices_and_merge_with_channel_dims(self):
        mask = checker_mask((2, 2), parity=True)
        # Simulate (H,W,C)
        x = jnp.stack(
            [
                jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            ],
            axis=-1,
        )  # shape (2,2,2)

        primary, secondary = mask.split(x, extra_channel_dims=1)
        reconstructed = mask.merge(primary, secondary, extra_channel_dims=1)
        np.testing.assert_allclose(reconstructed, x, rtol=RTOL)

    def test_rmul_rank_check_raises(self):
        mask = checker_mask((2, 2), parity=False)
        # rank(x) < len(event_shape) -> should raise
        msg = "rank too low for mask multiplication"
        with pytest.raises(ValueError, match=msg):
            _ = jnp.array([1.0, 2.0]) * mask


class TestAutoVmapReconstructor:
    class ParamEchoBijection(nnx.Module):
        def __init__(self, *, rngs=None):
            self.theta = nnx.Param(jnp.array(0.0))

        def forward(self, x, log_density):
            return x, log_density

        def reverse(self, x, log_density):
            return x, log_density

    def test_autovmap_attribute_passthrough_and_call(self):
        template = ModuleReconstructor(self.ParamEchoBijection())
        psize = template.params_total_size  # expect 1
        # Batched parameters over (batch, active)
        params = jnp.zeros((4, 3, psize))
        auto = template.from_params(params, autovmap=True)

        # Calling forward should accept input_ranks and be a no-op
        x = jnp.ones((4, 3))
        # ld must match x when input_ranks=(0,0) and param batch dims are (4,3)
        ld = jnp.zeros((4, 3))
        y, ld1 = auto.forward(x, ld, input_ranks=(0, 0))
        np.testing.assert_allclose(y, x)
        np.testing.assert_allclose(ld1, ld)


class TestModuleReconstructorFromState:
    def test_from_state_reconstruction(self, rng_key):
        base = AffineLinear(rngs=nnx.Rngs(rng_key))
        template = ModuleReconstructor(base)
        param_array = jnp.zeros((template.params_total_size,))
        # First reconstruct a module from array
        mod = template.from_params(param_array)
        # Extract params state from reconstructed module
        graph, state = nnx.split(mod)
        params_state, _ = nnx.split_state(state, nnx.Param, ...)
        # Reconstruct again using from_state
        mod2 = template.from_params(params_state)
        x = jnp.array([1.0, 2.0, 3.0])
        ld = jnp.array(0.0)
        y1, ld1 = mod.forward(x, ld)
        y2, ld2 = mod2.forward(x, ld)
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(ld1, ld2)


class TestShapeInvariance:
    @pytest.mark.parametrize("split", [True, False])
    @pytest.mark.parametrize("event_size", [1, 3])
    @pytest.mark.parametrize("bij_rank", [0])
    @pytest.mark.parametrize("parity", [True, False])
    @pytest.mark.parametrize("bs", [((),), ((5,),)])
    def test_basic_shapes(self, split, event_size, bij_rank, parity, bs, rng_key):
        # Deterministic check across shape variants without Hypothesis
        mask = checker_mask((event_size,), parity=parity)
        layer = TestGeneralCouplingLayer()._build_layer(
            mask, AffineLinear(rngs=nnx.Rngs(rng_key)), split=split, bij_rank=bij_rank
        )
        batch_shape = bs[0]
        x = jnp.ones(batch_shape + (event_size,))
        ld = jnp.zeros(batch_shape)
        y, ld1 = layer.forward(x, ld)
        assert y.shape == x.shape
        assert ld1.shape == ld.shape

    @pytest.mark.parametrize("split", [True, False])
    @pytest.mark.parametrize("parity", [True, False])
    @pytest.mark.parametrize("event_size", [3])
    @pytest.mark.parametrize("bij_rank", [0])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_shape_invariance_param_sweep(
        self, split, parity, event_size, bij_rank, batch_shape, rng_key
    ):
        mask = checker_mask((event_size,), parity=parity)
        layer = TestGeneralCouplingLayer()._build_layer(
            mask, AffineLinear(rngs=nnx.Rngs(rng_key)), split=split, bij_rank=bij_rank
        )
        x = jnp.ones(batch_shape + (event_size,))
        ld = jnp.zeros(batch_shape)
        y, ld1 = layer.forward(x, ld)
        assert y.shape == x.shape
        assert ld1.shape == ld.shape
