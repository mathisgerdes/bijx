"""
Focused tests for bijx.lie to improve coverage.

Covers:
- contract indices/trace
- scalar_prod and adjoint
- HaarDistribution API and shapes
- _isolate_argument
- grad() with return_value and has_aux
- value_grad_divergence()
- curve_grad() branches (value and jacfwd paths)
- path_grad(), path_grad2(), path_div()
"""

import jax
import jax.numpy as jnp
import numpy as np

import bijx.lie as lie


def _is_skew_hermitian(x: jnp.ndarray) -> bool:
    return jnp.allclose(lie.adjoint(x) + x, 0, atol=1e-10)


def _is_traceless(x: jnp.ndarray) -> bool:
    return jnp.allclose(jnp.trace(x), 0, atol=1e-10)


def _is_su(x: jnp.ndarray) -> bool:
    det = jnp.allclose(jnp.linalg.det(x), 1, atol=1e-10)
    self_adj = jnp.allclose(x @ lie.adjoint(x), lie.adjoint(x) @ x, atol=1e-10)
    return det and self_adj


class TestContractAndBasics:
    def test_contract_indices_and_trace(self):
        # Broadcasting across leading dims; chain multiplication
        a = jax.random.normal(jax.random.key(0), (2, 3, 4)).astype(jnp.complex64)
        b = jax.random.normal(jax.random.key(1), (1, 4, 5)).astype(jnp.complex64)
        c = jax.random.normal(jax.random.key(2), (2, 5, 3)).astype(jnp.complex64)

        res = lie.contract(a, b, c)
        # Indices API reachable and consistent type
        ind_in, ind_out = lie.contract(a, b, c, return_einsum_indices=True)
        assert isinstance(ind_in, str)
        assert isinstance(ind_out, str)
        # Sanity: output shape matches left-right chain
        # Manual chain with broadcasting
        res_ref = jnp.einsum("lma,lab->lmb", a, b)
        res_ref = jnp.einsum("lmb,lbc->lmc", res_ref, c)
        np.testing.assert_allclose(res, res_ref, atol=1e-6)

        # Trace
        x = jax.random.normal(jax.random.key(3), (2, 4, 4)).astype(jnp.complex64)
        y = jax.random.normal(jax.random.key(4), (2, 4, 4)).astype(jnp.complex64)
        tr = lie.contract(x, y, x, y, trace=True)
        assert tr.shape == (2,)

    def test_scalar_prod_orthonormal_su2(self):
        gens = lie.SU2_GEN
        gram = jnp.array(
            [[lie.scalar_prod(gens[i], gens[j]) for j in range(3)] for i in range(3)]
        )
        np.testing.assert_allclose(jnp.real(gram), jnp.eye(3), atol=1e-6)
        np.testing.assert_allclose(jnp.imag(gram), 0, atol=1e-6)

    def test_scalar_prod_orthonormal_su3(self):
        gens = lie.SU3_GEN
        gram = jnp.array(
            [[lie.scalar_prod(gens[i], gens[j]) for j in range(8)] for i in range(8)]
        )
        np.testing.assert_allclose(jnp.real(gram), jnp.eye(8), atol=1e-6)
        np.testing.assert_allclose(jnp.imag(gram), 0, atol=1e-6)

    def test_adjoint_batched_unitary(self, rng_key):
        u = lie.sample_haar(rng_key, n=2, batch_shape=(5,))
        ident = lie.adjoint(u) @ u
        target = jnp.broadcast_to(jnp.eye(2), ident.shape)
        np.testing.assert_allclose(ident, target, atol=1e-6)


class TestHaarDistribution:
    def test_sample_haar_basic(self, rng_key):
        key = rng_key
        su2 = lie.sample_haar(key, n=2)
        assert _is_su(su2)
        assert su2.shape == (2, 2)
        batch = lie.sample_haar(key, n=3, batch_shape=(5,))
        assert batch.shape == (5, 3, 3)
        assert _is_su(batch[0])

        su3 = lie.sample_haar(key, n=3)
        assert _is_su(su3)
        assert su3.shape == (3, 3)

    def test_distribution_shaped(self, rng_key):
        dist = lie.HaarDistribution(n=3, base_shape=(2, 2))
        samples, ld = dist.sample((4,), rng=rng_key)
        assert samples.shape == (4, 2, 2, 3, 3)
        assert ld.shape == (4,)
        np.testing.assert_allclose(ld, 0)

    def test_periodic_gauge_lattice(self, rng_key):
        dist = lie.HaarDistribution.periodic_gauge_lattice(n=3, lat_shape=(5, 5))
        samples, ld = dist.sample((7,), rng=rng_key)
        assert samples.shape == (7, 5, 5, 2, 3, 3)
        assert ld.shape == (7,)
        np.testing.assert_allclose(ld, 0)
        assert _is_su(samples.reshape(-1, 3, 3)[0])


class TestUtilities:
    def test_isolate_argument(self):
        def f(a, b, c):
            return a + b * c

        wrapped, orig_b = lie._isolate_argument(f, 1, 2.0, 3.0, 4.0)
        assert orig_b == 3.0
        assert wrapped(-1.0) == f(2.0, -1.0, 4.0)


class TestGradients:
    def test_grad_return_value_no_aux(self, rng_key):
        u = lie.sample_haar(rng_key, n=2)

        def loss(m):
            return jnp.trace(m).real

        val, g = lie.grad(loss, return_value=True, algebra=lie.SU2_GEN)(u)
        assert isinstance(val, jnp.ndarray) or np.isscalar(val)
        assert _is_skew_hermitian(g)
        assert _is_traceless(g)

    def test_grad_return_value_with_aux(self, rng_key):
        u = lie.sample_haar(rng_key, n=2)

        def loss_with_aux(m):
            return jnp.trace(m).real, jnp.array(0.0)

        val_aux, g = lie.grad(
            loss_with_aux,
            return_value=True,
            has_aux=True,
            algebra=lie.skew_traceless_cot,
        )(u)
        # val_aux is (value, aux)
        val, aux = val_aux
        assert jnp.ndim(val) == 0
        assert jnp.ndim(aux) == 0
        assert _is_skew_hermitian(g)
        assert _is_traceless(g)

    def test_value_grad_divergence(self, rng_key):
        u = lie.sample_haar(rng_key, n=2)

        def potential(m):
            return jnp.real(jnp.trace(m @ lie.adjoint(m)))

        val, g, lap = lie.value_grad_divergence(potential, u, lie.SU2_GEN)
        assert jnp.ndim(val) == 0
        assert _is_skew_hermitian(g)
        assert _is_traceless(g)
        assert jnp.isfinite(lap)

    def test_curve_grad_directions(self, rng_key):
        u = lie.sample_haar(rng_key, n=2)

        def loss(m):
            return jnp.trace(m).real

        full_grad = lie.grad(loss, algebra=lie.SU2_GEN)(u)
        direction = lie.SU2_GEN[0] * 0.5

        # jacfwd branch gives directional derivative at t=0
        dir_val = lie.curve_grad(loss, direction=direction)(u)
        np.testing.assert_allclose(dir_val, lie.scalar_prod(direction, full_grad))

        # value branch returns (value_at_t0, directional_derivative)
        v, dv = lie.curve_grad(loss, direction=direction, return_value=True)(u)
        np.testing.assert_allclose(v, loss(u))
        np.testing.assert_allclose(dv, dir_val)


class TestPathDerivatives:
    def test_path_grad_shapes(self, rng_key):
        u1 = lie.sample_haar(rng_key, n=2)
        u2 = lie.sample_haar(jax.random.split(rng_key)[0], n=2)

        def fun(us):
            m1, m2 = us
            return jnp.real(jnp.trace(m1 @ m2))

        out, jac_tree = lie.path_grad(fun, lie.SU2_GEN, [u1, u2])
        assert jnp.ndim(out) == 0
        assert isinstance(jac_tree, list)
        assert len(jac_tree) == 2
        for j_leaf, u in zip(jac_tree, [u1, u2]):
            # Components in generator basis per input
            assert j_leaf.shape == u.shape[:-2] + (len(lie.SU2_GEN),)

    def test_path_grad2_shapes(self, rng_key):
        u1 = lie.sample_haar(rng_key, n=2)
        u2 = lie.sample_haar(jax.random.split(rng_key)[0], n=2)

        def fun(us):
            m1, m2 = us
            return jnp.real(jnp.trace(m1 @ m2))

        out, jac, jac2 = lie.path_grad2(fun, lie.SU2_GEN, [u1, u2])
        assert jnp.ndim(out) == 0
        for tree in (jac, jac2):
            assert isinstance(tree, list)
            assert len(tree) == 2
            for leaf, u in zip(tree, [u1, u2]):
                assert leaf.shape == u.shape[:-2] + (len(lie.SU2_GEN),)

    def test_path_div_zero_vector_field(self, rng_key):
        u = lie.sample_haar(rng_key, n=2)

        def vf(_us):
            # single input: return generator components as vector
            return jnp.zeros((len(lie.SU2_GEN),), dtype=jnp.float32)

        out, div = lie.path_div(vf, lie.SU2_GEN, u)
        assert out.shape == (1, len(lie.SU2_GEN))
        np.testing.assert_allclose(div, 0.0)


class TestEigenvalueVisualization:
    """Tests for eigenvalue-based visualization functions."""

    def test_compute_haar_density_su2(self):
        """Test Haar density computation for SU(2)."""
        # SU(2) case: single angle
        angles = jnp.array([0.0, jnp.pi / 4, jnp.pi / 2])  # Shape (3,)
        angles_reshaped = angles[..., None]  # Shape (3, 1)
        density = lie.compute_haar_density(angles_reshaped)

        assert density.shape == (3,)
        # For SU(2), Haar density should be 4sin²(θ) = 4sin²(θ)
        # But our angles are actually 2*theta in the standard parameterization
        # So we get |e^(iθ) - e^(-iθ)|² = |2i sin(θ)|² = 4sin²(θ)
        expected = 4 * jnp.sin(angles) ** 2
        np.testing.assert_allclose(density, expected, atol=1e-6)

    def test_compute_haar_density_su3(self):
        """Test Haar density computation for SU(3)."""
        # SU(3) case: two angles
        angles = jnp.array([[0.1, 0.2], [0.5, -0.1]])  # Shape (2, 2)
        density = lie.compute_haar_density(angles)

        assert density.shape == (2,)
        assert jnp.all(density >= 0)  # Haar density should be non-negative
        assert jnp.all(jnp.isfinite(density))

    def test_create_eigenvalue_grid_shapes(self):
        """Test eigenvalue grid creation for different SU(N) groups."""
        # SU(2) case
        grid_2 = lie.create_eigenvalue_grid(n=2, grid_points=10)
        assert grid_2.shape == (10, 1)

        # SU(3) case
        grid_3 = lie.create_eigenvalue_grid(n=3, grid_points=5)
        assert grid_3.shape == (5, 5, 2)

        # SU(4) case
        grid_4 = lie.create_eigenvalue_grid(n=4, grid_points=3)
        assert grid_4.shape == (3, 3, 3, 3)

        # Check angle ranges for midpoint grid over [-π, π]
        expected_min = -jnp.pi + jnp.pi / 10
        expected_max = jnp.pi - jnp.pi / 10
        np.testing.assert_allclose(jnp.min(grid_2), expected_min, atol=1e-10)
        np.testing.assert_allclose(jnp.max(grid_2), expected_max, atol=1e-10)

    def test_construct_su_matrix_from_eigenvalues_unitarity(self, rng_key):
        """Test that constructed matrices are indeed SU(N)."""
        key = rng_key

        # Test SU(2)
        angles_2 = lie.create_eigenvalue_grid(n=2, grid_points=5)
        matrices_2 = lie.construct_su_matrix_from_eigenvalues(key, angles_2)

        assert matrices_2.shape == (5, 2, 2)

        # Check unitarity: U @ U† = I
        identity_check = matrices_2 @ matrices_2.conj().swapaxes(-1, -2)
        expected_identity = jnp.broadcast_to(jnp.eye(2), identity_check.shape)
        np.testing.assert_allclose(identity_check, expected_identity, atol=1e-6)

        # Check determinant = 1
        dets = jnp.linalg.det(matrices_2)
        np.testing.assert_allclose(dets, 1.0, atol=1e-6)

    def test_construct_su_matrix_from_eigenvalues_su3(self, rng_key):
        """Test SU(3) matrix construction."""
        key = rng_key

        # Test with simple angles
        angles = jnp.array([[0.1, 0.2], [0.5, -0.3]])
        matrices = lie.construct_su_matrix_from_eigenvalues(key, angles)

        assert matrices.shape == (2, 3, 3)

        # Check unitarity
        identity_check = matrices @ matrices.conj().swapaxes(-1, -2)
        expected_identity = jnp.broadcast_to(jnp.eye(3), identity_check.shape)
        np.testing.assert_allclose(identity_check, expected_identity, atol=1e-6)

        # Check determinant = 1
        dets = jnp.linalg.det(matrices)
        np.testing.assert_allclose(dets, 1.0, atol=1e-6)

    def test_construct_su_matrix_eigenvalue_consistency(self, rng_key):
        """Test that constructed matrices have the correct eigenvalue structure."""
        key = rng_key

        # For SU(2), if we give angle θ, eigenvalues should be e^(±iθ/2)
        # But our convention uses θ directly, so eigenvalues are e^(iθ), e^(-iθ)
        angles = jnp.array([[jnp.pi / 4]])  # Single SU(2) matrix
        matrices = lie.construct_su_matrix_from_eigenvalues(key, angles)

        # Get eigenvalues
        eigvals = jnp.linalg.eigvals(matrices[0])

        # Check magnitudes are 1 (on unit circle)
        np.testing.assert_allclose(jnp.abs(eigvals), 1.0, atol=1e-6)

        # Product of eigenvalues should be 1 (determinant constraint)
        np.testing.assert_allclose(jnp.prod(eigvals), 1.0, atol=1e-6)

        # Order-insensitive check via the trace: λ1 + λ2 = 2 cos(θ)
        theta = angles[0, 0]
        expected_tr = 2 * jnp.cos(theta)
        np.testing.assert_allclose(jnp.real(jnp.sum(eigvals)), expected_tr, atol=1e-6)
        np.testing.assert_allclose(jnp.imag(jnp.sum(eigvals)), 0.0, atol=1e-6)

    def test_evaluate_density_on_eigenvalue_grid(self, rng_key):
        """Test density evaluation on eigenvalue grids."""
        key = rng_key

        def simple_density(u):
            """Simple test density: Re[tr(U)]"""
            return jnp.real(jnp.trace(u, axis1=-2, axis2=-1))

        # Test SU(2)
        angles, density_vals, haar_weights = lie.evaluate_density_on_eigenvalue_grid(
            simple_density, n=2, grid_points=50, rng=key, normalization_domain="torus"
        )

        assert angles.shape == (50, 1)
        assert density_vals.shape == (50,)
        assert haar_weights.shape == (50,)
        assert jnp.all(haar_weights >= 0)  # Haar weights should be non-negative
        assert jnp.all(jnp.isfinite(density_vals))

        # Test normalization: coarse quadrature approximates 1
        volume_element = (2 * jnp.pi / 50) ** 1
        total_weight = jnp.sum(haar_weights * volume_element)
        np.testing.assert_allclose(total_weight, 1.0, rtol=0, atol=1e-3)

    def test_evaluate_density_on_eigenvalue_grid_su3(self, rng_key):
        """Test SU(3) density evaluation."""
        key = rng_key

        def identity_density(u):
            """Constant density function"""
            return jnp.ones(u.shape[:-2])

        angles, density_vals, haar_weights = lie.evaluate_density_on_eigenvalue_grid(
            identity_density,
            n=3,
            grid_points=20,
            rng=key,
            normalize=True,
            normalization_domain="torus",
        )

        assert angles.shape == (20**2, 2)  # 20^2 grid points flattened
        assert density_vals.shape == (20**2,)
        assert haar_weights.shape == (20**2,)

        # For constant density, the result should just be the normalized Haar weights
        np.testing.assert_allclose(density_vals, 1.0, atol=1e-6)

        # Check normalization via coarse quadrature
        volume_element = (2 * jnp.pi / 20) ** 2
        total_weight = jnp.sum(haar_weights * volume_element)
        np.testing.assert_allclose(total_weight, 1.0, rtol=0, atol=1e-3)

    def test_haar_eigenangle_normalization_constant(self):
        """Analytic constants: (2π)^(n-1) n! for SU(n)."""
        for n, expected in [(2, (2 * jnp.pi) ** 1 * 2), (3, (2 * jnp.pi) ** 2 * 6)]:
            c = lie._haar_eigenangle_normalization_constant(n, domain="torus")
            np.testing.assert_allclose(c, expected, rtol=1e-12, atol=0)

    def test_haar_density_mathematical_properties(self):
        """Test mathematical properties of Haar density."""
        # SU(2): density should be 4sin²(θ)
        angles = jnp.linspace(-jnp.pi + 0.1, jnp.pi - 0.1, 20)[..., None]
        density = lie.compute_haar_density(angles)

        # Check periodicity properties and positivity
        assert jnp.all(density >= 0)

        # Check that density is zero at θ = 0, ±π (where eigenvalues coincide)
        special_angles = jnp.array([[0.0], [jnp.pi], [-jnp.pi]])
        special_density = lie.compute_haar_density(special_angles)
        np.testing.assert_allclose(special_density, 0.0, atol=1e-10)

        # Maximum should be at θ = ±π/2
        max_angles = jnp.array([[jnp.pi / 2], [-jnp.pi / 2]])
        max_density = lie.compute_haar_density(max_angles)
        expected_max = 4.0  # 4sin²(π/2) = 4
        np.testing.assert_allclose(max_density, expected_max, atol=1e-10)
