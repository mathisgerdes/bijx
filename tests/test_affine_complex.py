"""Tests for complex affine bijections (``bijx.bijections.affine_complex``).

Covers the stateless ``complex_affine_apply`` kernel and the ``ComplexScaling``
bijection: forward formula, inverse round-trip, log-Jacobian correctness
(checked against the real-view Jacobian determinant), and ``complex_mask``
handling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from bijx import ComplexScaling
from bijx.bijections.affine_complex import complex_affine_apply

from .utils import ATOL, RTOL


def _real_logdet_forward(bijection, x):
    """log|det J| of the forward map, viewing complex x as a real 2n-vector."""
    n = x.size
    flat_shape = x.shape

    def fwd_real(v):
        xc = v[:n].reshape(flat_shape) + 1j * v[n:].reshape(flat_shape)
        y, _ = bijection.forward(xc, jnp.zeros(()))
        return jnp.concatenate([jnp.real(y).ravel(), jnp.imag(y).ravel()])

    v = jnp.concatenate([jnp.real(x).ravel(), jnp.imag(x).ravel()])
    jac = jax.jacobian(fwd_real)(v)
    _, logdet = jnp.linalg.slogdet(jac)
    return logdet


def _make(shape, *, with_phase=False, with_shift=False, seed=0):
    return ComplexScaling(
        shape,
        scale_init=nnx.initializers.normal(),
        phase_init=nnx.initializers.normal() if with_phase else None,
        shift_init=nnx.initializers.normal() if with_shift else None,
        rngs=nnx.Rngs(seed),
    )


class TestComplexAffineApply:
    """The stateless kernel."""

    def test_forward_matches_formula(self):
        x = jnp.array([1.0 + 2.0j, -0.5 + 0.3j])
        scale = jnp.array([2.0, 0.5])
        phase = jnp.array([0.3, -1.1])
        shift = jnp.array([1.0 + 0j, -1j])

        y, ld = complex_affine_apply(
            x,
            jnp.zeros(()),
            scale=scale,
            phase=phase,
            shift=shift,
            delta_ld=jnp.array(1.5),
        )

        expected = scale * jnp.exp(1j * phase) * x + shift
        np.testing.assert_allclose(y, expected, rtol=RTOL)
        # Forward subtracts the log-Jacobian contribution.
        np.testing.assert_allclose(ld, -1.5, atol=ATOL)

    def test_invert_is_exact_inverse(self):
        x = jnp.array([1.0 + 2.0j, -0.5 + 0.3j, 4.0 - 0.7j])
        scale = jnp.array([2.0, 0.5, 1.3])
        phase = jnp.array([0.3, -1.1, 2.0])
        shift = jnp.array([1.0 + 0j, -1j, 0.2 + 0.2j])
        delta = jnp.array(1.5)

        y, ld = complex_affine_apply(
            x, jnp.zeros(()), scale=scale, phase=phase, shift=shift, delta_ld=delta
        )
        x_back, ld_back = complex_affine_apply(
            y, ld, scale=scale, phase=phase, shift=shift, delta_ld=delta, invert=True
        )

        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, 0.0, atol=ATOL)


class TestComplexScaling:
    """The diagonal complex-affine bijection module."""

    @pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
    @pytest.mark.parametrize(
        ("with_phase", "with_shift"),
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_roundtrip(self, shape, with_phase, with_shift):
        bij = _make(shape, with_phase=with_phase, with_shift=with_shift)
        key = jax.random.key(0)
        x = jax.random.normal(key, (2, *shape), dtype=jnp.complex128)
        log_density = jnp.zeros(2)

        y, ld_fwd = bij.forward(x, log_density)
        x_back, ld_back = bij.reverse(y, ld_fwd)

        np.testing.assert_allclose(x_back, x, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(ld_back, log_density, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize(
        ("with_phase", "with_shift"),
        [(False, False), (True, False), (True, True)],
    )
    def test_log_density_matches_jacobian(self, with_phase, with_shift):
        shape = (3,)
        bij = _make(shape, with_phase=with_phase, with_shift=with_shift)
        x = jax.random.normal(jax.random.key(1), shape, dtype=jnp.complex128)

        _, ld = bij.forward(x, jnp.zeros(()))

        # bijx forward subtracts the log-Jacobian, so the reported change equals
        # -log|det J_forward| computed on the real 2n-dimensional view.
        expected = -_real_logdet_forward(bij, x)
        np.testing.assert_allclose(ld, expected, atol=ATOL, rtol=RTOL)

    def test_complex_mask_keeps_real_entries_real(self):
        # An all-zero mask marks every entry as real: no phase rotation and no
        # imaginary shift component should be applied, and the log-Jacobian
        # weight drops from 2 to 1 per entry.
        shape = (4,)
        mask = jnp.zeros(shape)
        bij = ComplexScaling(
            shape,
            scale_init=nnx.initializers.normal(),
            phase_init=nnx.initializers.normal(),
            shift_init=nnx.initializers.normal(),
            complex_mask=mask,
            rngs=nnx.Rngs(0),
        )

        x = jax.random.normal(jax.random.key(2), shape)  # real input
        y, ld = bij.forward(x, jnp.zeros(()))

        # Output stays real (no phase, no imaginary shift on masked entries).
        np.testing.assert_allclose(jnp.imag(y), 0.0, atol=ATOL)

        # Weight is 1 per entry -> delta_ld = sum(log_scale); reported = -delta_ld.
        log_scale = bij.scale.get_value()
        np.testing.assert_allclose(ld, -jnp.sum(log_scale), atol=ATOL, rtol=RTOL)
