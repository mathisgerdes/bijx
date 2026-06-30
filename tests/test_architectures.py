"""Tests for the example flow builders in ``bijx.architectures``.

Mirrors ``test_bijections_coupling.py``: bijectivity (round-trip), log-det vs
autodiff, batch/shape handling, and an init sanity (near-identity at init under
the chosen bias_mode) for ``coupling_flow``, ``realnvp_flow``,
``realnvp_conv_flow`` and the ``init_for`` presets.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import bijx

from .utils import assert_finite_and_real


def _spline_factory(rngs):
    return bijx.MonotoneRQSpline(8, (), rngs=rngs)


def _roundtrip(flow, x):
    ld0 = jnp.zeros(x.shape[:1]) if x.ndim > 1 else jnp.zeros(())
    y, ld = flow.forward(x, ld0)
    xb, ldb = flow.reverse(y, ld)
    return y, ld, xb, ldb, ld0


def _logdet_vs_autodiff(flow, x):
    """Compare reported forward log-det against the autodiff Jacobian determinant
    for a single (unbatched) event vector."""

    def fwd(xin):
        y, _ = flow.forward(xin, jnp.zeros(()))
        return y

    jac = jax.jacrev(fwd)(x)
    sign, logabsdet = jnp.linalg.slogdet(jac)
    _, reported = flow.forward(x, jnp.zeros(()))
    # bijx density convention: forward SUBTRACTS the log-Jacobian determinant
    # (ScalarBijection.forward uses -log_jac), so with input log-density 0 the
    # reported value equals -log|det J|.
    return float(reported), float(-logabsdet)


class TestCouplingFlow:
    @pytest.mark.parametrize("bias_mode", ["zeros", "extract_init"])
    @pytest.mark.parametrize("event_size", [3, 4])
    def test_roundtrip(self, bias_mode, event_size, rng_key):
        flow = bijx.coupling_flow(
            event_size,
            _spline_factory,
            n_coupling_layers=4,
            bias_mode=bias_mode,
            rngs=nnx.Rngs(rng_key),
        )
        x = jnp.linspace(-1.0, 1.0, 2 * event_size).reshape(2, event_size)
        y, ld, xb, ldb, ld0 = _roundtrip(flow, x)
        assert_finite_and_real(y, "coupling forward")
        np.testing.assert_allclose(xb, x, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(ldb, ld0, atol=1e-4, rtol=1e-4)

    def test_logdet_matches_autodiff(self, rng_key):
        flow = bijx.coupling_flow(
            4,
            _spline_factory,
            n_coupling_layers=3,
            bias_mode="zeros",
            rngs=nnx.Rngs(rng_key),
        )
        x = jnp.array([0.3, -0.5, 0.8, -0.1])
        reported, logabsdet = _logdet_vs_autodiff(flow, x)
        np.testing.assert_allclose(reported, logabsdet, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
    def test_batch_shapes(self, batch_shape, rng_key):
        flow = bijx.coupling_flow(
            3,
            _spline_factory,
            n_coupling_layers=2,
            bias_mode="zeros",
            rngs=nnx.Rngs(rng_key),
        )
        x = jnp.ones(batch_shape + (3,))
        ld = jnp.zeros(batch_shape)
        y, ld1 = flow.forward(x, ld)
        assert y.shape == x.shape
        assert ld1.shape == ld.shape

    def test_extract_init_near_identity(self, rng_key):
        # With a tiny delta init and extract_init bias, the flow starts near id.
        factory = bijx.init_for("cubic", architecture="coupling", depth=8)
        flow = bijx.coupling_flow(
            4,
            factory,
            n_coupling_layers=8,
            n_copies=2,
            bias_mode="extract_init",
            rngs=nnx.Rngs(rng_key),
        )
        x = jnp.linspace(-1.0, 1.0, 8).reshape(2, 4)
        y, ld = flow.forward(x, jnp.zeros(2))
        assert_finite_and_real(y, "near-identity forward")
        # With the extract_init bias seeding each element as a small-delta cubic
        # (a~1, b~0.3, beta~0 -> near identity per element), the deep flow stays
        # modest and well-conditioned at init (vs the O(10) blow-ups the study
        # saw with random scale/loc init). Bound the deviation and the log-det.
        assert float(jnp.max(jnp.abs(y - x))) < 2.0
        assert float(jnp.max(jnp.abs(ld))) < 5.0


class TestRealNVPFlow:
    def test_roundtrip(self, rng_key):
        flow = bijx.realnvp_flow(4, n_coupling_layers=4, rngs=nnx.Rngs(rng_key))
        x = jnp.linspace(-1.0, 1.0, 8).reshape(2, 4)
        y, ld, xb, ldb, ld0 = _roundtrip(flow, x)
        np.testing.assert_allclose(xb, x, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(ldb, ld0, atol=1e-5, rtol=1e-5)

    def test_identity_at_init(self, rng_key):
        # default zeros bias + zeros final kernel => exact identity
        flow = bijx.realnvp_flow(4, n_coupling_layers=4, rngs=nnx.Rngs(rng_key))
        x = jnp.linspace(-1.0, 1.0, 8).reshape(2, 4)
        y, ld = flow.forward(x, jnp.zeros(2))
        np.testing.assert_allclose(y, x, atol=1e-6)
        np.testing.assert_allclose(ld, 0.0, atol=1e-6)

    def test_logdet_matches_autodiff(self, rng_key):
        # perturb away from identity so the Jacobian is non-trivial
        flow = bijx.realnvp_flow(
            4,
            n_coupling_layers=3,
            rngs=nnx.Rngs(rng_key),
            final_kernel_init=nnx.initializers.normal(0.1),
        )
        x = jnp.array([0.3, -0.5, 0.8, -0.1])
        reported, logabsdet = _logdet_vs_autodiff(flow, x)
        np.testing.assert_allclose(reported, logabsdet, atol=1e-5, rtol=1e-5)


class TestRealNVPConvFlow:
    @pytest.mark.parametrize("channels", [1, 3])
    def test_roundtrip(self, channels, rng_key):
        flow = bijx.realnvp_conv_flow(
            (6, 6, channels), n_coupling_layers=2, rngs=nnx.Rngs(rng_key)
        )
        x = jnp.linspace(-1.0, 1.0, 2 * 6 * 6 * channels).reshape(2, 6, 6, channels)
        y, ld = flow.forward(x, jnp.zeros(2))
        xb, ldb = flow.reverse(y, ld)
        assert_finite_and_real(y, "conv forward")
        assert y.shape == x.shape
        assert ld.shape == (2,)
        np.testing.assert_allclose(xb, x, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(ldb, jnp.zeros(2), atol=1e-4, rtol=1e-4)


class TestInitFor:
    @pytest.mark.parametrize("bij", ["cubic", "cubic_rational", "sinh"])
    def test_stacked_factory_builds_and_roundtrips(self, bij, rng_key):
        factory = bijx.init_for(bij, architecture="stacked")
        gen = bijx.stack_bijections(factory, transform=bijx.ScanChain, copies=8)
        module = gen(nnx.Rngs(rng_key))
        x = jnp.linspace(-2.0, 2.0, 5)
        y, ld = module.forward(x, jnp.zeros(()))
        xb, ldb = module.reverse(y, ld)
        assert_finite_and_real(y, f"stacked {bij}")
        np.testing.assert_allclose(xb, x, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(("bij", "depth"), [("cubic", 8), ("cubic_rational", 16)])
    def test_coupling_factory_finite(self, bij, depth, rng_key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factory = bijx.init_for(bij, architecture="coupling", depth=depth)
            flow = bijx.coupling_flow(
                4,
                factory,
                n_coupling_layers=depth,
                n_copies=2,
                bias_mode="extract_init",
                rngs=nnx.Rngs(rng_key),
            )
        x = jnp.linspace(-1.0, 1.0, 8).reshape(2, 4)
        y, ld = flow.forward(x, jnp.zeros(2))
        xb, ldb = flow.reverse(y, ld)
        assert_finite_and_real(y, f"coupling {bij}")
        np.testing.assert_allclose(xb, x, atol=1e-3, rtol=1e-3)

    def test_cubic_rational_warns_below_depth_gate(self):
        with pytest.warns(UserWarning, match="depth >= ~16"):
            bijx.init_for("cubic_rational", architecture="coupling", depth=8)

    def test_sinh_coupling_warns_conservative_preset(self):
        with pytest.warns(UserWarning, match="conservative preset"):
            bijx.init_for("sinh", architecture="coupling", depth=8)

    def test_invalid_args(self):
        with pytest.raises(ValueError, match="bijection must be"):
            bijx.init_for("nope", architecture="coupling")
        with pytest.raises(ValueError, match="architecture must be"):
            bijx.init_for("cubic", architecture="nope")
