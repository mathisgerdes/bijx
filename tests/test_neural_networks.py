"""
Neural network component tests: convolutions, embeddings, features, simple nets.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bijx.nn.conv import (
    ConvSym,
    fold_kernel,
    kernel_d4,
    kernel_equidist,
    resize_kernel_weights,
    rot_lattice_90,
    unfold_kernel,
)
from bijx.nn.embeddings import (
    KernelFourier,
    KernelGauss,
    KernelLin,
    KernelReduced,
    PositionalEmbedding,
)
from bijx.nn.features import ConcatFeatures, FourierFeatures, PolynomialFeatures
from bijx.nn.nets import MLP, ConvNet, ResNet

from .utils import RTOL


class TestConvUtilities:
    def test_kernel_d4_orbits_3x3(self):
        n_orbits, orbits = kernel_d4((3, 3))
        # center, edges, corners
        assert n_orbits == 3
        assert orbits.shape == (3, 3)

    def test_kernel_equidist_basic(self):
        n_orbits, orbits = kernel_equidist((5, 5))
        assert n_orbits >= 3
        assert orbits.shape == (5, 5)
        # Equidistant orbits group sites by distance from the center, so the four
        # corners (equidistant) share one label and the center is its own label.
        corners = {
            int(orbits[0, 0]),
            int(orbits[0, 4]),
            int(orbits[4, 0]),
            int(orbits[4, 4]),
        }
        assert len(corners) == 1
        assert orbits[2, 2] not in corners

    def test_fold_unfold_consistency(self, rng_key):
        shape = (3, 3)
        n_orbits, orbits = kernel_d4(shape)
        in_c, out_c = 2, 3
        full = jax.random.normal(rng_key, shape + (in_c, out_c))
        folded = fold_kernel(full, orbits, n_orbits)
        unfolded = unfold_kernel(folded, orbits)
        # Unfolded weights are orbit-shared; check they are in fact the same
        for idx in range(n_orbits):
            group = unfolded[orbits == idx]  # shape: (num_sites, in_c, out_c)
            # Compare all entries to the first across the sites axis
            np.testing.assert_array_equal(
                group,
                # broadcasting adds orbit axis back
                jnp.broadcast_to(group[0], group.shape),
            )
        # Folding again should recover the same folded params
        refolded = fold_kernel(unfolded, orbits, n_orbits)
        np.testing.assert_array_equal(refolded, folded)

    def test_resize_kernel_weights_shape(self):
        k = jnp.ones((3, 3, 1, 1))
        k2 = resize_kernel_weights(k, (5, 5))
        assert k2.shape == (5, 5, 1, 1)

        # resizing back to the original shape should yield identity
        k3 = resize_kernel_weights(k, (3, 3))
        np.testing.assert_array_equal(k3, k)

    def test_resize_kernel_weights_same_shape_identity(self):
        # Resizing to the *same* shape must be the identity for every parity,
        # including even sizes (regression: even dims were halved at the edges).
        for shape in [(3, 3), (4, 4), (3, 4), (6, 5)]:
            k = jax.random.normal(jax.random.key(sum(shape)), shape + (2, 3))
            np.testing.assert_allclose(
                resize_kernel_weights(k, shape), np.asarray(k), rtol=RTOL
            )

    def _conv_with_kernel(self, kernel):
        # Build a ConvSym whose kernel equals `kernel` (orbit_function=None so
        # params are the raw, unfolded kernel), with no bias.
        ks = kernel.shape[:-2]
        in_c, out_c = kernel.shape[-2:]
        conv = ConvSym(
            in_c,
            out_c,
            kernel_size=ks,
            orbit_function=None,
            use_bias=False,
            rngs=nnx.Rngs(0),
        )
        conv.kernel_params = nnx.Param(jnp.asarray(kernel).reshape(-1, in_c, out_c))
        return conv

    def test_resize_kernel_weights_preserves_convolution(self):
        # The defining property: under ConvSym's CIRCULAR padding, enlarging a
        # kernel (zero-padding that preserves each tap's spatial offset) must
        # leave the convolution output on a periodic lattice unchanged.
        size = 11
        x1 = jax.random.normal(jax.random.key(0), (size, 1, 1))
        for kernel_size in (3, 4, 5, 6, 7):
            k = jax.random.normal(jax.random.key(kernel_size), (kernel_size, 1, 1))
            y0 = self._conv_with_kernel(k)(x1)
            for new in range(kernel_size, size + 1):
                k2 = resize_kernel_weights(k, (new,))
                y1 = self._conv_with_kernel(jnp.asarray(k2))(x1)
                np.testing.assert_allclose(y0, y1, atol=1e-5)

        size = 9
        x2 = jax.random.normal(jax.random.key(1), (size, size, 1, 1))
        for kernel_size in [(3, 3), (4, 4), (3, 5), (4, 6), (5, 4)]:
            k = jax.random.normal(
                jax.random.key(sum(kernel_size)), kernel_size + (1, 1)
            )
            y0 = self._conv_with_kernel(k)(x2)
            for new in [
                (kernel_size[0] + 1, kernel_size[1] + 1),
                (kernel_size[0] + 2, kernel_size[1] + 3),
                (size, size),
            ]:
                k2 = resize_kernel_weights(k, new)
                y1 = self._conv_with_kernel(jnp.asarray(k2))(x2)
                np.testing.assert_allclose(y0, y1, atol=1e-5)

    def test_resize_kernel_weights_shrink_drops_outer_taps(self):
        # Shrinking keeps the centered taps; if support lives in the center the
        # result must be conv-invariant under CIRCULAR padding.
        size = 11
        x1 = jax.random.normal(jax.random.key(3), (size, 1, 1))
        k = np.zeros((7, 1, 1))
        k[2:5, 0, 0] = np.asarray(jax.random.normal(jax.random.key(4), (3,)))
        y0 = self._conv_with_kernel(jnp.asarray(k))(x1)
        k3 = resize_kernel_weights(k, (3,))
        assert k3.shape == (3, 1, 1)
        y1 = self._conv_with_kernel(jnp.asarray(k3))(x1)
        np.testing.assert_allclose(y0, y1, atol=1e-5)

    def test_rot_lattice_90_four_times_identity(self):
        x = jnp.arange(3 * 3).reshape(3, 3)
        y = x
        for _ in range(3):
            y = rot_lattice_90(y, 0, 1)
            # make sure rot_lattice_90 is not identity
            assert not jnp.allclose(y, x)
        y = rot_lattice_90(y, 0, 1)
        np.testing.assert_array_equal(y, x)


class TestConvSym:
    def test_param_shapes_and_forward(self, rng_key):
        r = nnx.Rngs(rng_key)
        conv_sym = ConvSym(1, 2, kernel_size=(3, 3), orbit_function=kernel_d4, rngs=r)
        conv_none = ConvSym(1, 2, kernel_size=(3, 3), orbit_function=None, rngs=r)
        # Parameter storage shapes differ under symmetry vs none
        n_orbits, _ = kernel_d4((3, 3))
        assert conv_sym.kernel_params.shape == (n_orbits, 1, 2)
        assert conv_none.kernel_params.shape == (9, 1, 2)
        # Forward shape
        x = jnp.ones((8, 8, 1))
        y = conv_sym(x)
        assert y.shape == (8, 8, 2)

    def test_call_with_external_kernel_params(self, rng_key):
        # Passing kernel_params overrides the layer's own; None reproduces the default,
        # and a supplied (generated) kernel is used instead and is differentiable.
        r = nnx.Rngs(rng_key)
        conv = ConvSym(1, 2, kernel_size=(3, 3), orbit_function=kernel_d4, rngs=r)
        x = jnp.ones((6, 6, 1))
        # default path equals explicitly passing the layer's own params
        y_default = conv(x)
        y_explicit = conv(x, kernel_params=conv.kernel_params[...])
        assert jnp.allclose(y_default, y_explicit)
        # a different (generated) kernel changes the output and carries gradients back
        gen = jnp.ones_like(conv.kernel_params[...])
        y_gen = conv(x, kernel_params=gen)
        assert not jnp.allclose(y_default, y_gen)
        g = jax.grad(lambda kp: jnp.sum(conv(x, kernel_params=kp)))(gen)
        assert jnp.any(g != 0)

    def test_grad_through_params(self, rng_key):
        conv = ConvSym(1, 1, kernel_size=(3, 3), rngs=nnx.Rngs(rng_key))
        x = jnp.ones((5, 5, 1))

        def loss_fn(params, variables, graph):
            # Reconstruct module from params and static graph
            model = nnx.merge(graph, params, variables)
            y = model(x)
            return jnp.mean((y - 0.5) ** 2)

        # Extract params and compute gradient
        graph, params, variables = nnx.split(conv, nnx.Param, ...)
        val, grads = jax.value_and_grad(loss_fn)(params, variables, graph)
        assert jnp.isfinite(val)

        # Sum of squares across array leaves should be positive
        def _accumulate(acc, x):
            return acc + jnp.sum(x**2)

        total = jax.tree_util.tree_reduce(_accumulate, grads, 0.0)
        assert total > 0.0


class TestEmbeddings:
    def test_kernel_gauss_shape_and_norm(self, rng_key):
        emb = KernelGauss(21, adaptive_width=True, norm=True, rngs=nnx.Rngs(rng_key))
        out = emb(0.3)
        assert out.shape == (21,)
        np.testing.assert_allclose(out.sum(), 1.0, rtol=RTOL)

    def test_kernel_lin_shape(self):
        emb = KernelLin(11)
        # Use column vector to enable broadcasting against feature axis
        out = emb(0.1)
        assert out.shape == (11,)

        out = emb(jnp.linspace(0.0, 1.0, 10))
        assert out.shape == (10, 11)

    def test_kernel_fourier_shape_and_const(self):
        # Use odd feature_count
        emb = KernelFourier(21)
        out = emb(0.2)
        assert out.shape == (21,)

        out = emb(jnp.linspace(0.0, 1.0, 10))
        assert out.shape == (10, 21)

    def test_kernel_reduced(self, rng_key):
        base = KernelFourier(21)
        red = KernelReduced(base, 8, rngs=nnx.Rngs(rng_key))
        out = red(0.5)
        assert out.shape == (8,)

        out = red(jnp.linspace(0.0, 1.0, 10))
        assert out.shape == (10, 8)

    def test_positional_embedding_shapes(self):
        emb = PositionalEmbedding(64, append_input=True)
        vals = jnp.linspace(0.0, 1.0, 5)
        out = emb(vals)
        assert out.shape == (5, 64 + 1)

        out = emb(jnp.linspace(0.0, 1.0, 10))
        assert out.shape == (10, 64 + 1)


class TestFeatures:
    def test_fourier_features_divergence(self, rng_key):
        feats = FourierFeatures(
            feature_count=8, input_channels=1, rngs=nnx.Rngs(rng_key)
        )
        x = jnp.ones((4, 4, 1))
        y, div_map = feats(x)
        assert y.shape == (4, 4, 8)
        local = jnp.ones((feats.feature_count, 1))
        div = div_map(local)
        assert jnp.all(jnp.isfinite(div))

    def test_polynomial_features_basic(self, rng_key):
        feats = PolynomialFeatures([1, 2, 3], input_channels=1, rngs=nnx.Rngs(rng_key))
        # Non-trivial input pins the actual powers: feature_k(x) = x**p_k.
        x = jnp.full((2, 2, 1), 2.0)
        y, div_map = feats(x)
        assert y.shape == (2, 2, 3)
        expected = jnp.broadcast_to(jnp.array([2.0, 4.0, 8.0]), y.shape)
        np.testing.assert_allclose(y, expected, rtol=RTOL)
        local = jnp.ones((len(feats.powers), 1))
        div = div_map(local)
        assert jnp.all(jnp.isfinite(div))

    def test_concat_features(self, rng_key):
        r = nnx.Rngs(rng_key)
        f1 = FourierFeatures(4, input_channels=1, rngs=r)
        f2 = PolynomialFeatures([1, 2], input_channels=1, rngs=r)
        combo = ConcatFeatures([f1, f2], rngs=r)
        x = jnp.ones((3, 3, 1))
        y, _ = combo(x)
        assert y.shape == (3, 3, 6)
        # Concatenation must equal stacking the sub-feature outputs along channels.
        y1, _ = f1(x)
        y2, _ = f2(x)
        np.testing.assert_allclose(y, jnp.concatenate([y1, y2], axis=-1), rtol=RTOL)


class TestSimpleNets:
    def test_convnet_shapes(self, rng_key):
        net = ConvNet(
            in_channels=1, out_channels=2, kernel_size=(3, 3), rngs=nnx.Rngs(rng_key)
        )
        x = jnp.ones((10, 10, 1))
        y = net(x)
        assert y.shape == (10, 10, 2)

    def test_resnet_and_mlp_shapes_and_grads(self, rng_key):
        r = nnx.Rngs(rng_key)
        res = ResNet(in_features=16, out_features=8, width=32, depth=2, rngs=r)
        mlp = MLP(in_features=8, out_features=4, hidden_features=[16, 8], rngs=r)
        x = jnp.ones((5, 16))
        y = res(x)
        assert y.shape == (5, 8)
        z = mlp(y)
        assert z.shape == (5, 4)

        # Gradient w.r.t. inputs to ensure backward path exists
        def loss_on_x(xin):
            out = mlp(res(xin))
            return jnp.mean((out - 0.1) ** 2)

        g = jax.grad(loss_on_x)(x)
        assert jnp.isfinite(jnp.sum(g))
