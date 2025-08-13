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
        assert conv_sym.kernel_params.value.shape == (n_orbits, 1, 2)
        assert conv_none.kernel_params.value.shape == (9, 1, 2)
        # Forward shape
        x = jnp.ones((8, 8, 1))
        y = conv_sym(x)
        assert y.shape == (8, 8, 2)

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
        # Local coupling diagonal per channel and per feature
        local = jnp.ones((1, 1, feats.feature_count))
        y, div = feats(x, local_coupling=local)
        assert y.shape == (4, 4, 8)
        assert jnp.all(jnp.isfinite(div))

    def test_polynomial_features_basic(self, rng_key):
        feats = PolynomialFeatures([1, 2, 3], input_channels=1, rngs=nnx.Rngs(rng_key))
        x = jnp.ones((2, 2, 1))
        local = jnp.ones((1, 1, len(feats.powers)))
        y, div = feats(x, local_coupling=local)
        assert y.shape == (2, 2, 3)
        assert jnp.all(jnp.isfinite(div))

    def test_concat_features(self, rng_key):
        r = nnx.Rngs(rng_key)
        f1 = FourierFeatures(4, input_channels=1, rngs=r)
        f2 = PolynomialFeatures([1, 2], input_channels=1, rngs=r)
        combo = ConcatFeatures([f1, f2], rngs=r)
        x = jnp.ones((3, 3, 1))
        # Sum of features across components
        local = jnp.ones((1, 1, f1.feature_count + len(f2.powers)))
        y, _ = combo(x, local_coupling=local)
        assert y.shape == (3, 3, 6)


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
