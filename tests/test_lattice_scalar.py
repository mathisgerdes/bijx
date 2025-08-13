"""
Scalar lattice utilities split from test_physics for focus and clarity.
"""

import jax.numpy as jnp
import numpy as np

from bijx.lattice.scalar import (
    correlation_length,
    cyclic_corr,
    cyclic_tensor,
    kinetic_term,
    phi4_term,
    phi4_term_alt,
    poly_term,
    two_point,
    two_point_central,
)


class TestLatticeScalarUtilities:
    def test_cyclic_corr_vs_tensor_mean(self):
        a = jnp.arange(6.0).reshape(2, 3)
        b = jnp.flip(a, axis=0)
        c1 = cyclic_corr(a, b)
        c2 = jnp.mean(cyclic_tensor(a, b), axis=(-2, -1))
        np.testing.assert_allclose(c1, c2)

    def test_two_point_and_central_shapes(self):
        phis = jnp.ones((5, 4, 4))
        g = two_point(phis)
        gc = two_point_central(phis)
        assert g.shape == (4, 4)
        assert gc.shape == (4, 4)

    def test_correlation_length_finite(self):
        x = jnp.exp(-jnp.arange(16.0).reshape(4, 4) / 3.0)
        xi = correlation_length(x)
        assert jnp.isfinite(xi)

    def test_phi4_terms_and_kinetic(self):
        phi = jnp.ones((3, 3))
        kin = kinetic_term(phi)
        np.testing.assert_allclose(kin, 0.0)
        a = phi4_term(phi, m2=1.0, lam=0.5)
        a_alt = phi4_term_alt(phi, kappa=0.0, lam=0.5)
        exp_alt = (1 - 2 * 0.5) * phi**2 + 0.5 * phi**4
        np.testing.assert_allclose(a_alt, exp_alt)
        assert jnp.all(a >= 0)
        v = poly_term(phi, jnp.array([1.0, 0.0]), even=True)
        np.testing.assert_allclose(v, phi**0 * 1.0)
