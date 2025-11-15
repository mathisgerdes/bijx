r"""
Continuous normalizing flows with ODE-based vector fields.

This module implements continuous normalizing flows (CNFs), which use neural ODEs
to define bijective transformations through the integration of time-dependent
vector fields.

Key components:
- :class:`ContFlowDiffrax`: CNF using diffrax for advanced ODE solving
- :class:`ContFlowRK4`: CNF using fixed-step RK4 integration
- :class:`ContFlowCG`: CNF using Crouch-Grossmann integration for Lie groups
- :class:`AutoJacVF`: Automatic computation of vector field Jacobian traces

Mathematical background:
Continuous normalizing flows define bijections through ODEs of the form:
$$\frac{d\mathbf{z}}{dt} = f(\mathbf{z}(t), t; \theta)$$

The log-density transforms according to the instantaneous change of variables:
$$\frac{d \log p}{dt} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$$

This allows exact likelihood computation for continuous transformations.
"""

import typing as tp
from functools import partial

import diffrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax_autovmap import autovmap

from .. import cg
from ..solvers import DiffraxConfig, odeint_rk4
from ..utils import ShapeInfo
from .base import Bijection


class ContFlowDiffrax(Bijection):
    r"""Continuous normalizing flow using diffrax ODE solver.

    Wraps around a vector field to turn it into the bijection defined by
    solving the corresponding ODE, using the diffrax library.
    The vector field function should return both the velocity and the
    log-density time derivative for the instantaneous change of variables.

    Args:
        vf: Vector field module with signature
            ``(t, x, **kwargs) -> (dx/dt, d(log_density)/dt)``.
        config: Hyperparameters (end times, solver, etc.) passed to diffrax.

    Example:
        >>> # Define vector field
        >>> vf = SomeVectorField()
        >>> config = DiffraxConfig(solver=diffrax.Tsit5(), dt=0.1)
        >>> flow = ContFlowDiffrax(vf, config)
        >>> y, log_det = flow.forward(x, log_density)

    Important:
        The vector field should generally be a callable ``nnx.Module``.
        However, it cannot mutate it's internal state variables
        (batch averages, counting number of calls, etc.)
        as that is incompatible with the internal ODE solver.
    """

    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: nnx.Module,
        config: DiffraxConfig = DiffraxConfig(),
    ):
        self.vf_graph, vf_variables, vf_meta = nnx.split(vf, nnx.Variable, ...)
        self.vf_variables = nnx.data(vf_variables)
        self.vf_meta = nnx.data(vf_meta)
        self.config = config
        # assert config.saveat == diffrax.SaveAt(t1=True), "saveat must be t1=True"

    def _vf(self, t, state, args):
        """Vector field wrapper for diffrax integration."""
        variables, kwargs = args
        return nnx.merge(self.vf_graph, variables, self.vf_meta)(t, state[0], **kwargs)

    def solve_flow(
        self,
        x,
        log_density,
        *,
        # integration parameters
        t_start: float | None = None,
        t_end: float | None = None,
        dt: float | None = None,
        saveat: diffrax.SaveAt | None = None,
        # arguments to vector field
        **kwargs,
    ):
        """Solve the ODE flow with optional parameter overrides.

        Args:
            x: Initial state array.
            log_density: Initial log density values.
            t_start: Override integration start time.
            t_end: Override integration end time.
            dt: Override step size.
            saveat: Override save configuration.
            **kwargs: Additional arguments passed to vector field.

        Returns:
            Diffrax solution object containing integration results.
        """
        config = self.config.optional_override(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            saveat=saveat,
        )
        term = diffrax.ODETerm(self._vf)
        y0 = (x, log_density)
        sol = config.solve(term, y0, (self.vf_variables, kwargs))
        return sol

    def forward(self, x, log_density, **kwargs):
        """Solve the ODE flow and return the final state.

        Note that the same optional overrides as in :meth:`solve_flow`
        can be set here (start/end times, step size).
        The ``saveat`` argument should probably not be modified as the
        this method assumes the final state is computed, which is returned by
        this function. Use :meth:`solve_flow` directly for this.
        """
        sol = self.solve_flow(x, log_density, **kwargs)
        return jax.tree.map(lambda x: x[-1], sol.ys)

    def reverse(self, x, log_density, **kwargs):
        """Solve the ODE flow in reverse and return the final state.

        Note that the same optional overrides as in :meth:`solve_flow`
        can be set here (start/end times, step size).
        The ``saveat`` argument should probably not be modified as the
        this method assumes the final state is computed, which is returned by
        this function. Use :meth:`solve_flow` directly for this.
        """
        sol = self.solve_flow(
            x,
            log_density,
            t_start=self.config.t_end,
            t_end=self.config.t_start,
            **kwargs,
        )
        return jax.tree.map(lambda x: x[-1], sol.ys)


class ContFlowRK4(Bijection):
    r"""Continuous normalizing flow using fixed-step RK4 solver.

    Wraps around a vector field to turn it into the bijection defined by
    solving the corresponding ODE, using a fixed-step RK4 solver.
    The vector field function should return both the velocity and the
    log-density time derivative for the instantaneous change of variables.

    The integration uses a uniform time grid with fixed step size.
    Gradients are always computed using backward solving (adjoint sensitivity).
    Consider :class:`ContFlowDiffrax` for more flexibility and advanced solvers.

    Args:
        vf: Vector field function with signature
            ``(t, x, **kwargs) -> (dx/dt, d(log_density)/dt)``.
        t_start: Integration start time.
        t_end: Integration end time.
        steps: Number of integration steps.

    Example:
        >>> def vector_field(t, x):
        ...     return -x, jnp.sum(x, axis=-1, keepdims=True)  # Linear flow
        >>> flow = ContFlowRK4(vector_field, steps=50)
        >>> y, log_det = flow.forward(x, log_density)
    """

    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        *,
        t_start: float = 0,
        t_end: float = 1,
        steps: int = 20,
    ):
        self.vf = vf
        self.t_start = t_start
        self.t_end = t_end
        self.steps = steps

    def solve_flow(
        self,
        x,
        log_density,
        *,
        # integration parameters
        t_start=None,
        t_end=None,
        steps=None,
        # arguments to vector field
        **kwargs,
    ):
        """Solve the ODE flow using RK4 integration.

        Args:
            x: Initial state array.
            log_density: Initial log density values.
            t_start: Override integration start time.
            t_end: Override integration end time.
            steps: Override number of integration steps.
            **kwargs: Additional arguments passed to vector field.

        Returns:
            Final state tuple (x_final, log_density_final).
        """
        t_start = t_start if t_start is not None else self.t_start
        t_end = t_end if t_end is not None else self.t_end
        steps = steps if steps is not None else self.steps

        delta_t = t_end - t_start
        sgn = jnp.where(delta_t < 0, -1.0, 1.0)

        def vf(s, state, args):
            x, log_density = state
            t = t_start + s * delta_t
            dx_dt, dld_dt = jax.tree.map(
                lambda x: sgn * x,
                self.vf(t, x, **args),
            )
            return dx_dt, dld_dt

        y0 = (x, log_density)
        y_final = odeint_rk4(
            vf,
            y0,
            1.0,
            kwargs,
            step_size=1.0 / steps,  # cannot be a jax tracer here
            start_time=0.0,
        )
        return y_final

    def forward(self, x, log_density, **kwargs):
        return self.solve_flow(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.solve_flow(
            x,
            log_density,
            t_start=self.t_end,
            t_end=self.t_start,
            **kwargs,
        )


class ContFlowCG(Bijection):
    r"""Continuous normalizing flow using Crouch-Grossmann integration.

    The state can be any pytree containing leaves of real/complex arrays
    or matrix group elements.
    See :mod:`bijx.cg` for more details.

    Args:
        vf: Vector field function with signature
            ``(t, x, **kwargs) -> (dx/dt, d(log_density)/dt)``.
        t_start: Integration start time.
        t_end: Integration end time.
        steps: Number of integration steps.
        tableau: Butcher tableau specifying the integration scheme.

    Example:
        >>> # Vector field for SU(N) gauge field evolution
        >>> flow = ContFlowCG(SomeGaugeVF(), tableau=cg.CG3)
        >>> U_final, log_det = flow.forward(U, log_density)
    """

    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        # default to single gauge object
        x_type: tp.Any = cg.Unitary(),
        *,
        t_start: float = 0,
        t_end: float = 1,
        steps: int = 20,
        tableau: cg.ButcherTableau = cg.CG2,
    ):
        self.vf = vf
        self.x_type = x_type
        self.t_start = t_start
        self.t_end = t_end
        self.steps = steps
        self.tableau = tableau

    def solve_flow(
        self,
        x,
        log_density,
        *,
        # integration parameters
        t_start=None,
        t_end=None,
        steps=None,
        # arguments to vector field
        **kwargs,
    ):
        """Solve the ODE flow using Crouch-Grossmann integration.

        Args:
            x: Initial state array.
            log_density: Initial log density values.
            t_start: Override integration start time.
            t_end: Override integration end time.
            steps: Override number of integration steps.
            **kwargs: Additional arguments passed to vector field.

        Returns:
            Final state tuple (x_final, log_density_final).
        """
        t_start = t_start if t_start is not None else self.t_start
        t_end = t_end if t_end is not None else self.t_end
        steps = steps if steps is not None else self.steps

        dt = (t_end - t_start) / steps

        def vf(t, state, args):
            x, log_density = state
            dx_dt, dld_dt = self.vf(t, x, **args)
            return dx_dt, dld_dt

        y0 = (x, log_density)

        y_final = cg.crouch_grossmann(
            vf,
            y0,
            kwargs,
            t_start,
            t_end,
            step_size=dt,
            manifold_types=(self.x_type, cg.Euclidean()),
            args_types=cg.Euclidean(),
            tableau=self.tableau,
        )
        return y_final

    def forward(self, x, log_density, **kwargs):
        return self.solve_flow(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.solve_flow(
            x,
            log_density,
            t_start=self.t_end,
            t_end=self.t_start,
            **kwargs,
        )


def _ndim_jacobian(func, event_dim):
    """Compute Jacobian matrix for multi-dimensional vector field.

    Creates a function that computes both the vector field value and its
    full Jacobian matrix using forward-mode automatic differentiation.

    Args:
        func: Vector field function to differentiate.
        event_dim: Number of event dimensions for shape processing.

    Returns:
        Function that returns (field_value, jacobian_matrix).
    """

    info = ShapeInfo(event_dim=event_dim)

    def func_flat(x_flat, event_shape):
        x = x_flat.reshape(x_flat.shape[:-1] + event_shape)
        out = func(x)
        return out.reshape(out.shape[: -len(event_shape)] + (-1,))

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, -1))
    def _jvp(x, tang):
        x_flat, _, _info = info.process_and_flatten(x)
        _func = partial(func_flat, event_shape=_info.event_shape)
        v, jac = jax.jvp(_func, (x_flat,), (tang,))
        return v.reshape(v.shape[:-1] + _info.event_shape), jac

    @autovmap(event_dim)
    def call_and_jac(x):
        _, _info = info.process_event(jnp.shape(x))
        tang_basis = jnp.eye(_info.event_size)
        v, jac = _jvp(x, tang_basis)
        return v, jac

    return call_and_jac


class AutoJacVF(nnx.Module):
    r"""Automatic Jacobian computation for vector fields.

    Wraps a vector field to automatically compute the trace of its Jacobian,
    which is needed for the log-density time derivative in continuous normalizing flows.
    This eliminates the need to manually implement the divergence computation,
    but is generally inefficient in higher dimensions.

    Args:
        vector_field_base: Base vector field ``(t, x, **kwargs) -> dx/dt``.
        event_dim: Number of event dimensions for proper shape handling.

    Example:
        >>> auto_vf = AutoJacVF(lambda t, x: -x, event_dim=1)
        >>> velocity, log_density_rate = auto_vf(t, x)
    """

    def __init__(self, vector_field_base, event_dim=1):
        self.vector_field_base = vector_field_base
        self.event_dim = event_dim

    def __call__(self, t, x, **kwargs):
        """Compute vector field and automatic Jacobian trace.

        Args:
            t: Time parameter.
            x: State array.

        Returns:
            Tuple of (velocity, negative_jacobian_trace).
        """
        jac_fn = _ndim_jacobian(
            partial(self.vector_field_base, t, **kwargs), self.event_dim
        )
        v, jac = jac_fn(x)
        return v, -jnp.trace(jac, axis1=-2, axis2=-1)
