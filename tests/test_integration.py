import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxcg import CG2, CG3, EULER, crouch_grossmann


class TestCG(unittest.TestCase):
    def test_sigma3(self):
        y0 = np.eye(2, dtype=complex)
        sigma_3 = np.array([[1, 0], [0, -1]])

        # gives vector field as element of lie algebra
        def vf(t, y, _):
            return 1j * sigma_3

        for tb in [EULER, CG2, CG3]:
            t1 = np.random.uniform(0, 2*np.pi)
            steps = 100

            y1 = crouch_grossmann(vf, y0, None, 0.0, t1, t1/steps, True, tableau=tb)
            a = np.exp(t1 * 1j * sigma_3) * y0
            close = np.allclose(y1, a, rtol=0, atol=2e-5)
            self.assertTrue(close, f'integration for t1={t1}, tableau {tb}')

    def test_backprop(self):
        y0 = jnp.eye(2, dtype=complex)
        sigma_3 = jnp.array([[1, 0], [0, -1]])

        # function giving vector at identity, to be transported to define vector field

        def vectorfield(t, y, theta=1.0):
            return 1j * theta * sigma_3

        theta_goal = 2.5
        y_goal = jnp.array([[jnp.exp(1j * theta_goal), 0], [0, jnp.exp(-1j * theta_goal)]])

        def loss(y):
            return jnp.sum(jnp.abs(y - y_goal)**2)

        dt = 1/100
        def loss_fn(params):
            y1 = crouch_grossmann(vectorfield, y0, params, 0, 1, dt, True, CG2)
            return loss(y1)

        @jax.jit
        @jax.vmap
        def automatic(params):
            return jax.grad(loss_fn)(params)

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None))
        def numeric(params, numeric_stepsize):
            return (loss_fn(params + numeric_stepsize) - loss_fn(params)) / numeric_stepsize

        thetas = jnp.linspace(-4, 4, 100)
        auto, num = automatic(thetas), numeric(thetas, .01)
        close = jnp.allclose(auto, num, rtol=.1, atol=.05)
        self.assertTrue(close)


if __name__ == '__main__':
    unittest.main()