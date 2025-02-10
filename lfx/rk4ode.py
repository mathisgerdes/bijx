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

from functools import partial

import jax
import jax.numpy as jnp
from jax import core, custom_derivatives
from jax.experimental.ode import ravel_first_arg
from jax.flatten_util import ravel_pytree


# partially following
# https://github.com/google/jax/blob/main/jax/experimental/ode.py


def odeint_rk4(fun, y0, end_time, *args, step_size, start_time=0):
    """Fixed step-size Runge-Kutta implementation.

    Args:
        fun: Function (t, y, *args) -> dy/dt giving the time derivative at
            the current position y and time t. The output must have the same
            shape and type as `y0`.
        y0: Initial value.
        end_time: Final time of the integration.
        *args: Additional arguments for `func`.
        step_size: Step size for the fixed-grid solver.
        start_time: Initial time of the integration.
    Returns:
        Final value `y` after the integration,
        of the same shape and type as `y0`.
    """
    # use other convention below...
    def _fun(y, t, *args):
        return fun(t, y, *args)

    for arg in jax.tree_util.tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f'The contents of odeint *args must be arrays or scalars, but got {arg}.')
    ts = jnp.array([start_time, end_time], dtype=float)

    converted, consts = custom_derivatives.closure_convert(_fun, y0, ts[0], *args)
    return _odeint_grid_wrapper(converted, step_size, y0, ts, *args, *consts)


@partial(jax.jit, static_argnums=(0, 1))
def _odeint_grid_wrapper(fun, step_size, y0, ts, *args):
    y0, unravel = ravel_pytree(y0)
    fun = ravel_first_arg(fun, unravel)
    out = _rk4_odeint(fun, step_size, y0, ts, *args)
    return unravel(out)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _rk4_odeint(fun, step_size, y0, ts, *args):
    func_ = lambda y, t: fun(y, t, *args)

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
    t0_bar = 0.
    y_bar = g

    # Compute effect of moving measurement time
    t_bar = jnp.dot(fun(y_final, ts[1], *args), g)
    t0_bar = t0_bar - t_bar

    # Run augmented system backwards
    _, y_bar, t0_bar, args_bar = odeint_rk4(
        aug_dynamics, (y_final, y_bar, t0_bar, args_bar),
        -ts[0], *args, step_size=step_size, start_time=-ts[1])

    ts_bar = jnp.array([t0_bar, t_bar])
    return (y_bar, ts_bar, *args_bar)


_rk4_odeint.defvjp(_rk4_odeint_fwd, _rk4_odeint_rev)
