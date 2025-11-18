r"""
ODE solvers for continuous normalizing flows.

This module provides ODE integration utilities specifically designed for
continuous normalizing flows, including custom RK4 implementations and
diffrax configuration helpers. The solvers support both forward and
reverse integration with automatic differentiation.
"""

from dataclasses import dataclass, replace
from functools import partial

import diffrax
import flax.typing as ftp
import jax
import jax.numpy as jnp
from flax import nnx
from jax import core, custom_derivatives
from jax.experimental.ode import api_util, ravel_first_arg
from jax.flatten_util import ravel_pytree

__all__ = [
    "DiffraxConfig",
    "odeint_rk4",
]


@dataclass(frozen=True)
class DiffraxConfig(nnx.Pytree):
    """Configuration for diffrax ODE solving in continuous normalizing flows.

    Encapsulates all parameters needed for diffrax-based ODE integration,
    including solver choice, step size control, and adjoint method selection.
    Provides convenient parameter override functionality for runtime configuration.

    Args:
        solver: Diffrax solver instance (default: Tsit5 adaptive solver).
        t_start: Integration start time.
        t_end: Integration end time.
        dt: (Initial) Step size for integration.
        saveat: Configuration for which time points to save.
        stepsize_controller: Strategy for adaptive step size control.
        adjoint: Adjoint method for gradient computation.
        event: Optional event detection during integration.
        max_steps: Maximum number of integration steps allowed.
        throw: Whether to raise exceptions on integration failure.
        solver_state: Initial solver internal state.
        controller_state: Initial step size controller state.
        made_jump: Whether the solver has made a discontinuous jump.

    Note:
        For more information, see [diffrax's documentation](https://docs.kidger.site/diffrax/).

    Example:
        >>> config = DiffraxConfig(
        ...     solver=diffrax.Dopri5(),
        ...     dt=0.1,
        ...     adjoint=diffrax.RecursiveCheckpointAdjoint()
        ... )
    """

    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.05
    saveat: nnx.Data[diffrax.SaveAt] = diffrax.SaveAt(t1=True)
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
    adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()
    event: diffrax.Event | None = None
    max_steps: int | None = 4096
    throw: bool = True
    solver_state: ftp.ArrayPytree | None = None
    controller_state: ftp.ArrayPytree | None = None
    made_jump: bool | None = None

    def replace(self, **changes):
        """Create new config with specified parameters replaced."""
        return replace(self, **changes)

    def optional_override(
        self,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        dt: float | None = None,
        saveat: diffrax.SaveAt | None = None,
        solver_state: ftp.ArrayPytree | None = None,
        controller_state: ftp.ArrayPytree | None = None,
    ):
        """Create new config with optionally overridden parameters.

        Args:
            t_start: Override integration start time.
            t_end: Override integration end time.
            dt: Override step size.
            saveat: Override save configuration.
            solver_state: Override solver internal state.
            controller_state: Override step size controller state.

        Returns:
            New DiffraxConfig with specified parameters overridden.
        """
        config = self
        if t_start is not None:
            config = config.replace(t_start=t_start)
        if t_end is not None:
            config = config.replace(t_end=t_end)
        if dt is not None:
            config = config.replace(dt=dt)
        if saveat is not None:
            config = config.replace(saveat=saveat)
        if solver_state is not None:
            config = config.replace(solver_state=solver_state)
        if controller_state is not None:
            config = config.replace(controller_state=controller_state)

        return config

    def solve(self, terms, y0, args):
        """Solve ODE using configured diffrax solver.

        Args:
            terms: Diffrax ODE terms defining the vector field.
            y0: Initial condition.
            args: Additional arguments passed to the vector field.

        Returns:
            Diffrax solution object containing integration results.
        """
        dt = jnp.abs(self.dt) * jnp.sign(self.t_end - self.t_start)

        return diffrax.diffeqsolve(
            terms,
            self.solver,
            t0=self.t_start,
            t1=self.t_end,
            dt0=dt,
            y0=y0,
            args=args,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            event=self.event,
            max_steps=self.max_steps,
            throw=self.throw,
            solver_state=self.solver_state,
            controller_state=self.controller_state,
            made_jump=self.made_jump,
        )


# odeint_rk4 is a modified version of the original odeint_rk4 function in
# jax.experimental.ode.
#
# Adapted from https://github.com/google/jax/blob/main/jax/experimental/ode.py
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def odeint_rk4(fun, y0, end_time, *args, step_size, start_time=0):
    """Fixed step-size Runge-Kutta implementation with custom adjoint.

    Provides a lightweight RK4 integrator optimized for continuous normalizing
    flows. Includes custom backward pass implementation using the adjoint method
    for efficient gradient computation in neural ODE applications.

    Args:
        fun: Function ``(t, y, *args) -> dy/dt`` giving the time derivative at
            the current position y and time t. The output must have the same
            shape and type as `y0`.
        y0: Initial value.
        end_time: Final time of the integration.
        ``*args``: Additional arguments for `func`.
        step_size: Step size for the fixed-grid solver.
        start_time: Initial time of the integration.

    Returns:
        Final value `y` after the integration, of the same shape and type as `y0`.

    Note:
        The custom VJP implementation uses the adjoint method, integrating
        backwards in time to compute gradients efficiently. This is particularly
        important for neural ODEs where the forward pass can be very long.

    Example:
        >>> def vector_field(t, y):
        ...     return -y  # Simple exponential decay
        >>> y_final = odeint_rk4(vector_field, 1.0, 1.0, step_size=0.01)
    """

    # use other convention below...
    def _fun(y, t, *args):
        return fun(t, y, *args)

    for arg in jax.tree_util.tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of *args must be arrays or scalars, but got {arg}."
            )
    ts = jnp.array([start_time, end_time], dtype=float)

    converted, consts = custom_derivatives.closure_convert(_fun, y0, ts[0], *args)
    return _odeint_grid_wrapper(converted, step_size, y0, ts, *args, *consts)


@partial(jax.jit, static_argnums=(0, 1))
def _odeint_grid_wrapper(fun, step_size, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    fun = ravel_first_arg(fun, unravel, api_util.debug_info("odeint", fun, args, {}))
    out = _rk4_odeint(fun, step_size, y0, ts, *args)
    return unravel(out)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _rk4_odeint(fun, step_size, y0, ts, *args):

    def func_(y, t):
        return fun(y, t, *args)

    def step_func(cur_t, dt, cur_y):
        """Take one step of RK4."""
        k1 = func_(cur_y, cur_t)
        k2 = func_(cur_y + dt * k1 / 2, cur_t + dt / 2)
        k3 = func_(cur_y + dt * k2 / 2, cur_t + dt / 2)
        k4 = func_(cur_y + dt * k3, cur_t + dt)
        return (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

    def cond_fun(carry):
        """Check if we've reached the last timepoint."""
        cur_y, cur_t = carry
        return cur_t < ts[1]

    def body_fun(carry):
        """Take one step of RK4."""
        cur_y, cur_t = carry
        next_t = jnp.minimum(cur_t + step_size, ts[1])
        dt = next_t - cur_t
        dy = step_func(cur_t, dt, cur_y)
        return cur_y + dy, next_t

    init_carry = (y0, ts[0])
    y1, t1 = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return y1


def _rk4_odeint_fwd(fun, step_size, y0, ts, *args):
    y_final = _rk4_odeint(fun, step_size, y0, ts, *args)
    return y_final, (y_final, ts, args)


def _rk4_odeint_rev(fun, step_size, res, g):
    y_final, ts, args = res

    def aug_dynamics(t, augmented_state, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        # `t` here is negative time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(fun, y, -t, *args)
        return (-y_dot, *vjpfun(y_bar))

    args_bar = jax.tree_util.tree_map(jnp.zeros_like, args)
    t0_bar = 0.0
    y_bar = g

    # Compute effect of moving measurement time
    t_bar = jnp.dot(fun(y_final, ts[1], *args), g)
    t0_bar = t0_bar - t_bar

    # Run augmented system backwards
    _, y_bar, t0_bar, args_bar = odeint_rk4(
        aug_dynamics,
        (y_final, y_bar, t0_bar, args_bar),
        -ts[0],
        *args,
        step_size=step_size,
        start_time=-ts[1],
    )

    ts_bar = jnp.array([t0_bar, t_bar])
    return (y_bar, ts_bar, *args_bar)


_rk4_odeint.defvjp(_rk4_odeint_fwd, _rk4_odeint_rev)
