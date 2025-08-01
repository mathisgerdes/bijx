import unittest
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from bijx.lie import SU2_GEN, U1_GEN, liegrad, sample_haar


class Potential(nn.Module):
    layers: tuple[int, ...] = (64, 64)
    traceless: bool = True
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, y):

        x = jnp.stack([y.real, y.imag], 0).reshape(1, -1)
        for l in self.layers:
            x = self.activation(nn.Dense(l)(x))

        init = nn.initializers.normal(1e-1)
        x = nn.Dense(1, kernel_init=init)(x)
        return x.reshape(())


class TestGrad(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.rng = jax.random.PRNGKey(0)

        self.pot1 = Potential()
        self.u1 = jnp.exp(1j * .45).reshape(1, 1)
        self.p1 = self.pot1.init(self.next_rng(), self.u1)

        self.pot2 = Potential()
        self.u2 = jax.scipy.linalg.expm(.1 * SU2_GEN[0])
        self.p2 = self.pot2.init(self.next_rng(), self.u2)

    def next_rng(self):
        self.rng, rng = jax.random.split(self.rng)
        return rng

    def test_u1(self):
        grad1 = liegrad.grad(self.pot1.apply, 1, True, algebra=U1_GEN)
        curve_grad = lambda p, u, g: liegrad.path_grad(partial(self.pot1.apply, p), g, u)

        v1, g1 = grad1(self.p1, self.u1)
        self.assertFalse(jnp.allclose(g1, 0))  # check it's not trivial
        v2, c2 = curve_grad(self.p1, self.u1, U1_GEN)
        # values should be the same
        self.assertTrue(jnp.allclose(v1, v2))
        g2 = jnp.einsum('i,ijk->jk', c2, U1_GEN)
        self.assertTrue(jnp.allclose(g1, g2))


    def test_su2(self):
        grad1 = liegrad.grad(self.pot1.apply, 1, True, algebra=SU2_GEN)
        grad2 = liegrad.grad(self.pot1.apply, 1, True, algebra=liegrad.skew_traceless_cot)
        curve_grad = lambda p, u, g: liegrad.path_grad(partial(self.pot2.apply, p), g, u)

        v1, g1 = grad1(self.p2, self.u2)
        v2, g2 = grad2(self.p2, self.u2)

        v3, c3 = curve_grad(self.p2, self.u2, SU2_GEN)
        g3 = jnp.einsum('i,ijk->jk', c3, SU2_GEN)

        self.assertFalse(jnp.allclose(g1, 0))  # check it's not trivial

        self.assertTrue(jnp.allclose(v1, v2))
        self.assertTrue(jnp.allclose(v1, v3))

        self.assertTrue(jnp.allclose(g1, g2))
        self.assertTrue(jnp.allclose(g1, g3))

    def test_divergence(self):
        v1, g1, d1 = liegrad.value_grad_divergence(partial(self.pot2.apply, self.p2), self.u2, SU2_GEN)

        # can also compute things via curve gradients
        def component_grad(u, gen):
            return liegrad.curve_grad(self.pot2.apply, gen, 1)(self.p2, u)

        def div(gen):
            grad = liegrad.curve_grad(component_grad, gen, 0, return_value=True)
            return grad(self.u2, gen)

        grads, divs = jax.vmap(div)(SU2_GEN)
        vect = jnp.einsum('i,i...->i...', grads, SU2_GEN)

        g2 = jnp.sum(vect, axis=0)
        d2 = jnp.sum(divs, axis=0)

        self.assertTrue(jnp.allclose(g1, g2))
        self.assertTrue(jnp.allclose(d1, d2))

    def test_vec_grad(self):
        def loop(us):
            return jnp.trace(us[0] @ us[1].conj().T @ us[0] @ us[2])

        us = sample_haar(self.rng, 2, count=3)

        def loop_comp(u, comp=0):
            _us = us.at[comp].set(u)
            return jnp.trace(_us[0] @ _us[1].conj().T @ _us[0] @ _us[2]).real

        # always real if SU(2)
        imag = loop(us).imag
        self.assertTrue(jnp.allclose(imag, 0.0, atol=1e-7))

        val, grad, grad2 = liegrad.path_grad2(loop, SU2_GEN, us)

        for i in range(len(us)):
            fn = partial(loop_comp, comp=i)

            v, g = liegrad.grad(fn, return_value=True)(us[i])
            self.assertTrue(jnp.allclose(v, val, atol=1e-6))
            grad_vec = jnp.einsum('a,aij->...ij', grad[i], SU2_GEN)
            self.assertTrue(jnp.allclose(grad_vec, g, atol=1e-6, rtol=1e-6))

            v, g, d = liegrad.value_grad_divergence(fn, us[i], SU2_GEN)

            self.assertTrue(jnp.allclose(v, val, atol=1e-6))
            g = jnp.einsum('a,aij->...ij', grad[i], SU2_GEN)
            self.assertTrue(jnp.allclose(grad_vec, g, atol=1e-6, rtol=1e-6))

            div = jnp.sum(grad2[i])
            self.assertTrue(jnp.allclose(div, d, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
