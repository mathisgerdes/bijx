"""
Tests for integration methods and samplers in bijx.

This module tests:
- Crouch-Grossmann integrators for Lie group ODEs (extensive testing)
- ODE solver interfaces (minimal API coverage)
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

import bijx.lie as lie

# Import components to test
from bijx import cg
from bijx.cg import CG2, CG3, EULER, crouch_grossmann
from bijx.solvers import DiffraxConfig, odeint_rk4

# Import test utilities
from .utils import assert_finite_and_real


class TestCrouchGrossmann:
    """Tests for Crouch-Grossmann integrators."""

    def test_butcher_tableau_creation(self):
        """Test creation and properties of Butcher tableaux."""
        # Test predefined tableaux
        assert EULER.stages == 1
        assert CG2.stages == 2
        assert CG3.stages == 3

        # Check consistency conditions for CG2
        # c_i = sum_j a_{i,j}
        for i in range(CG2.stages):
            c_expected = sum(CG2.a[i])
            assert abs(CG2.c[i] - c_expected) < 1e-15

        # sum_i b_i = 1
        assert abs(sum(CG2.b) - 1.0) < 1e-15

        # Check explicit property (a_{i,j} = 0 for j >= i)
        for i in range(CG2.stages):
            for j in range(i, CG2.stages):
                assert CG2.a[i][j] == 0

    def test_scalar_ode_exponential_decay(self):
        """Test CG integration on simple scalar exponential decay."""

        # ODE: dx/dt = -λx, exact solution: x(t) = x0 * exp(-λt)
        def decay_vf(t, x, args):
            return -args["lambda"] * x

        x0 = 2.0
        lambda_val = 0.5
        args = {"lambda": lambda_val}
        t0, t1 = 0.0, 2.0

        # Test different tableaux
        for tableau in [EULER, CG2, CG3]:
            # Use fine time step for accuracy
            step_size = 0.01
            x_final = crouch_grossmann(
                decay_vf,
                x0,
                args,
                t0,
                t1,
                step_size,
                manifold_types=cg.Euclidean(),
                tableau=tableau,
            )

            # Compare with exact solution
            x_exact = x0 * jnp.exp(-lambda_val * (t1 - t0))

            # Higher order methods should be more accurate
            if tableau == EULER:
                rtol = 1e-2
            elif tableau == CG2:
                rtol = 1e-5
            else:  # CG3
                rtol = 1e-8

            np.testing.assert_allclose(x_final, x_exact, rtol=rtol)

    def test_su2_exponential_known_result(self):
        """Test SU(2) evolution with known analytic result."""

        # Use sigma_3 generator: Y' = i*sigma_3*Y, solution Y(t) = exp(i*t*sigma_3)*Y_0
        def su2_vf(t, y, args):
            return args["strength"] * lie.SU2_GEN[2]  # sigma_3 (already has 1j factor)

        y0 = jnp.eye(2, dtype=complex)
        strength = 1.5
        args = {"strength": strength}
        t0, t1 = 0.0, 1.0

        for tableau in [EULER, CG2, CG3]:
            y_final = crouch_grossmann(
                su2_vf,
                y0,
                args,
                t0,
                t1,
                0.05,
                manifold_types=cg.Unitary(special=True),
                tableau=tableau,
            )

            # Exact solution
            y_exact = jax.scipy.linalg.expm(strength * (t1 - t0) * lie.SU2_GEN[2]) @ y0

            # Check unitarity is preserved exactly (up to numerical precision)
            np.testing.assert_allclose(
                y_final @ y_final.conj().T, jnp.eye(2), atol=1e-10
            )

            # Check determinant = 1
            np.testing.assert_allclose(jnp.linalg.det(y_final), 1.0, atol=1e-10)

            # Compare with exact solution
            rtol = 1e-3 if tableau == EULER else 1e-6

            np.testing.assert_allclose(y_final, y_exact, rtol=rtol, atol=1e-8)

    def test_so3_rotation_preservation(self):
        """Test SO(3) evolution preserves rotation group properties."""

        def so3_vf(t, r, args):
            # Constant angular velocity around z-axis
            omega_vec = args["omega"] * jnp.array([0.0, 0.0, 1.0])
            # Convert to skew-symmetric matrix (so(3) element)
            return jnp.array(
                [
                    [0, -omega_vec[2], omega_vec[1]],
                    [omega_vec[2], 0, -omega_vec[0]],
                    [-omega_vec[1], omega_vec[0], 0],
                ]
            )

        r0 = jnp.eye(3)
        omega = 2.0
        args = {"omega": omega}

        r_final = crouch_grossmann(
            so3_vf, r0, args, 0.0, 1.0, 0.1, manifold_types=cg.Matrix(), tableau=CG3
        )

        # Check orthogonality: R^T R = I
        np.testing.assert_allclose(r_final.T @ r_final, jnp.eye(3), atol=1e-12)

        # Check determinant = 1
        np.testing.assert_allclose(jnp.linalg.det(r_final), 1.0, atol=1e-12)

        # Check against exact solution (rotation around z-axis)
        angle = omega * 1.0
        r_exact = jnp.array(
            [
                [jnp.cos(angle), -jnp.sin(angle), 0],
                [jnp.sin(angle), jnp.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        np.testing.assert_allclose(r_final, r_exact, rtol=1e-6)

    def test_mixed_system_euclidean_lie(self):
        """Test mixed system with both Euclidean and Lie group components."""

        def mixed_vf(t, state, args):
            x, _ = state["position"], state["rotation"]

            # Euclidean dynamics: damped motion
            dx_dt = -args["damping"] * x

            # Lie group dynamics: constant angular velocity
            omega_vec = args["omega"] * jnp.array([1.0, 0.0, 0.0])
            dr_dt = jnp.array(
                [
                    [0, -omega_vec[2], omega_vec[1]],
                    [omega_vec[2], 0, -omega_vec[0]],
                    [-omega_vec[1], omega_vec[0], 0],
                ]
            )

            return {"position": dx_dt, "rotation": dr_dt}

        # Initial state
        state0 = {"position": jnp.array([1.0, 2.0, 0.5]), "rotation": jnp.eye(3)}

        # Geometry specification - pytree of manifold types
        manifold_types = {"position": cg.Euclidean(), "rotation": cg.Matrix()}

        args = {"damping": 0.5, "omega": 1.0}

        state_final = crouch_grossmann(
            mixed_vf,
            state0,
            args,
            0.0,
            2.0,
            0.1,
            manifold_types=manifold_types,
            tableau=CG2,
        )

        # Check Euclidean component decays exponentially
        x_exact = state0["position"] * jnp.exp(-args["damping"] * 2.0)
        np.testing.assert_allclose(state_final["position"], x_exact, rtol=1e-3)

        # Check rotation matrix properties
        r_final = state_final["rotation"]
        np.testing.assert_allclose(r_final.T @ r_final, jnp.eye(3), atol=1e-12)
        np.testing.assert_allclose(jnp.linalg.det(r_final), 1.0, atol=1e-12)

    def test_gradient_flow_through_integration(self):
        """Test gradient computation through CG integration."""

        def parameterized_vf(t, y, args):
            # Parameterized vector field on SU(2)
            return args["strength"] * lie.SU2_GEN[0]

        def loss_fn(strength):
            args = {"strength": strength}
            y_final = crouch_grossmann(
                parameterized_vf,
                jnp.eye(2, dtype=complex),
                args,
                0.0,
                1.0,
                0.1,
                manifold_types=cg.Unitary(special=True),
                tableau=CG2,
            )
            # Simple loss: trace of final state
            return jnp.real(jnp.trace(y_final))

        # Test gradient computation
        grad_fn = jax.grad(loss_fn)
        strength = 0.5
        grad_val = grad_fn(strength)

        assert_finite_and_real(jnp.array(grad_val), "CG gradient")

        # Verify with numerical gradient
        eps = 1e-5
        grad_numerical = (loss_fn(strength + eps) - loss_fn(strength - eps)) / (2 * eps)
        np.testing.assert_allclose(grad_val, grad_numerical, rtol=1e-3)

    def test_solver_consistency_comparison(self):
        """Test convergence properties and consistency between different tableaux."""

        def oscillatory_growth_vf(t, x, args):
            """ODE: dx/dt = sin(t) * x, solution: x(t) = x0 * exp(1 - cos(t))."""
            return jnp.sin(t) * x

        x0 = 1.0
        args = {}
        t0, t1 = 0.0, 1.0

        # Analytical solution for comparison
        x_exact = x0 * jnp.exp(1.0 - jnp.cos(t1))

        # Test convergence with multiple decreasing step sizes
        step_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125]
        errors_euler = []
        errors_cg2 = []
        errors_cg3 = []

        for step_size in step_sizes:
            x_euler = crouch_grossmann(
                oscillatory_growth_vf,
                x0,
                args,
                t0,
                t1,
                step_size,
                manifold_types=cg.Euclidean(),
                tableau=EULER,
            )
            x_cg2 = crouch_grossmann(
                oscillatory_growth_vf,
                x0,
                args,
                t0,
                t1,
                step_size,
                manifold_types=cg.Euclidean(),
                tableau=CG2,
            )
            x_cg3 = crouch_grossmann(
                oscillatory_growth_vf,
                x0,
                args,
                t0,
                t1,
                step_size,
                manifold_types=cg.Euclidean(),
                tableau=CG3,
            )

            errors_euler.append(abs(x_euler - x_exact))
            errors_cg2.append(abs(x_cg2 - x_exact))
            errors_cg3.append(abs(x_cg3 - x_exact))

        # Test convergence: errors should decrease as step size decreases
        for i in range(1, len(errors_euler)):
            assert errors_euler[i] < errors_euler[i - 1], "Euler should converge"
            assert errors_cg2[i] < errors_cg2[i - 1], "CG2 should converge"
            assert errors_cg3[i] < errors_cg3[i - 1], "CG3 should converge"

        # Test convergence rates: higher order methods should be more accurate
        # For the finest step size, verify order relationships
        finest_idx = -1
        assert (
            errors_cg2[finest_idx] < errors_euler[finest_idx]
        ), "CG2 should be more accurate than Euler"
        assert (
            errors_cg3[finest_idx] < errors_cg2[finest_idx]
        ), "CG3 should be more accurate than CG2"

        # Test theoretical convergence rates.
        if len(step_sizes) >= 4:
            # Test convergence order using Richardson extrapolation approach
            # Compare errors at step sizes h and h/2
            h_large = step_sizes[-3]
            h_small = step_sizes[-2]
            step_ratio = h_large / h_small  # Should be 2.0

            if abs(step_ratio - 2.0) < 0.1:  # Step sizes approximately halve
                # Calculate observed convergence orders: error(h) ≈ C * h^p
                # So log(error(h)/error(h/2)) ≈ p * log(2)
                euler_order = jnp.log(errors_euler[-3] / errors_euler[-2]) / jnp.log(
                    step_ratio
                )
                cg2_order = jnp.log(errors_cg2[-3] / errors_cg2[-2]) / jnp.log(
                    step_ratio
                )
                cg3_order = jnp.log(errors_cg3[-3] / errors_cg3[-2]) / jnp.log(
                    step_ratio
                )

                # Verify convergence orders (with some tolerance for numerical effects)
                assert euler_order > 0.5, f"Euler order {euler_order:.2f} should be ~1"
                assert cg2_order > 1.5, f"CG2 order {cg2_order:.2f} should be ~2"
                assert cg3_order > 2.0, f"CG3 order {cg3_order:.2f} should be ~3"

                # Higher-order methods should have higher convergence rates
                assert (
                    cg2_order > euler_order
                ), "CG2 should have higher convergence order than Euler"
                assert (
                    cg3_order > cg2_order
                ), "CG3 should have higher convergence order than CG2"

        # Fallback test: simple error ratio comparison
        else:
            # Compare error ratios between consecutive step sizes
            euler_ratio = errors_euler[-2] / errors_euler[-1]
            cg2_ratio = errors_cg2[-2] / errors_cg2[-1]
            cg3_ratio = errors_cg3[-2] / errors_cg3[-1]

            # Higher-order methods should have larger error reduction ratios
            assert (
                cg2_ratio > euler_ratio * 0.8
            ), "CG2 should converge faster than Euler"
            assert cg3_ratio > cg2_ratio * 0.8, "CG3 should converge faster than CG2"


class TestODESolvers:
    """Tests for ODE solver interfaces."""

    def test_diffrax_config_basic(self):
        """Test basic DiffraxConfig functionality."""
        config = DiffraxConfig(t_start=0.0, t_end=1.0, dt=0.1, solver=diffrax.Euler())

        # Test parameter override
        overridden = config.optional_override(t_end=2.0, dt=0.05)
        assert overridden.t_end == 2.0
        assert overridden.dt == 0.05
        assert overridden.t_start == 0.0  # unchanged

    def test_odeint_rk4_basic(self):
        """Test basic RK4 solver API."""

        def simple_ode(t, y, args):
            return -0.5 * y  # exponential decay

        y0 = jnp.array([2.0])

        # Test integration
        y_final = odeint_rk4(simple_ode, y0, 1.0, {}, step_size=0.1, start_time=0.0)

        assert_finite_and_real(y_final, "RK4 result")
        assert y_final.shape == y0.shape

        # Should decay towards zero
        assert y_final[0] < y0[0]

        # Compare with exact solution
        y_exact = y0 * jnp.exp(-0.5 * 1.0)
        np.testing.assert_allclose(y_final, y_exact, rtol=1e-3)

    def test_solver_consistency_rk4_vs_cg(self):
        """Test that RK4 and CG give similar results for Euclidean ODEs."""

        def ode_fn(t, y, args):
            return -0.2 * y + 0.1 * jnp.sin(t)

        y0 = jnp.array([1.0])
        args = {}

        # RK4 solution
        y_rk4 = odeint_rk4(ode_fn, y0, 2.0, args, step_size=0.01)

        # CG solution (with Euclidean manifold type)
        y_cg = crouch_grossmann(
            ode_fn, y0, args, 0.0, 2.0, 0.01, manifold_types=cg.Euclidean(), tableau=CG2
        )

        # Should give similar results
        np.testing.assert_allclose(y_rk4, y_cg, rtol=1e-5)


class TestIntegrationConsistency:
    """Integration tests checking consistency between different solvers."""

    def test_exponential_decay_all_solvers(self):
        """Test that all solvers agree on simple exponential decay."""

        # Simple ODE with known solution
        def decay_ode(t, y, args):
            return -args["rate"] * y

        y0 = jnp.array([1.0])
        args = {"rate": 0.3}
        t_final = 1.0

        # Exact solution
        y_exact = y0 * jnp.exp(-args["rate"] * t_final)

        # Test different solvers
        y_rk4 = odeint_rk4(decay_ode, y0, t_final, args, step_size=0.01)
        y_cg_euler = crouch_grossmann(
            decay_ode,
            y0,
            args,
            0.0,
            t_final,
            0.01,
            manifold_types=cg.Euclidean(),
            tableau=EULER,
        )
        y_cg2 = crouch_grossmann(
            decay_ode,
            y0,
            args,
            0.0,
            t_final,
            0.01,
            manifold_types=cg.Euclidean(),
            tableau=CG2,
        )

        # All should be reasonably close to exact solution
        np.testing.assert_allclose(y_rk4, y_exact, rtol=1e-4)
        np.testing.assert_allclose(y_cg_euler, y_exact, rtol=1e-2)  # Lower order
        np.testing.assert_allclose(y_cg2, y_exact, rtol=1e-4)

        # Higher order methods should be more accurate
        error_rk4 = jnp.abs(y_rk4 - y_exact)
        error_euler = jnp.abs(y_cg_euler - y_exact)
        error_cg2 = jnp.abs(y_cg2 - y_exact)

        assert error_cg2 < error_euler
        assert error_rk4 < error_euler


class TestNumericalStability:
    """Tests for numerical edge cases and stability."""

    def test_cg_zero_step_size(self):
        """Test CG behavior with zero step size."""

        def dummy_vf(t, y, args):
            return y  # Any vector field

        y0 = jnp.array([1.0])

        # Zero time interval should return initial state
        y_final = crouch_grossmann(
            dummy_vf, y0, {}, 0.0, 0.0, 0.1, manifold_types=cg.Euclidean(), tableau=CG2
        )
        np.testing.assert_allclose(y_final, y0, atol=1e-15)

    def test_large_time_integration(self):
        """Test stability over large time intervals."""

        def stable_oscillator(t, y, args):
            # Hamiltonian system: conserves energy
            x, p = y
            return jnp.array([p, -x])  # dx/dt = p, dp/dt = -x

        y0 = jnp.array([1.0, 0.0])  # Initial position=1, momentum=0

        # Integrate for long time with CG (should preserve energy)
        y_final = crouch_grossmann(
            stable_oscillator,
            y0,
            {},
            0.0,
            10.0,
            0.1,
            manifold_types=cg.Euclidean(),
            tableau=CG2,
        )

        # Check energy conservation: E = 0.5*(x^2 + p^2)
        energy_initial = 0.5 * jnp.sum(y0**2)
        energy_final = 0.5 * jnp.sum(y_final**2)

        # Should conserve energy reasonably well
        np.testing.assert_allclose(energy_final, energy_initial, rtol=1e-2)
        np.testing.assert_allclose(energy_final, energy_initial, rtol=1e-2)
        np.testing.assert_allclose(energy_final, energy_initial, rtol=1e-2)
