"""
Continuous-time vector fields.
"""

import typing as tp
from functools import partial

import diffrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax_autovmap import auto_vmap

from .. import cg
from ..solvers import DiffraxConfig, odeint_rk4
from ..utils import ShapeInfo
from .base import Bijection


class ContFlowDiffrax(Bijection):
    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: nnx.Module,
        config: DiffraxConfig = DiffraxConfig(),
    ):
        self.vf_graph, self.vf_variables, self.vf_meta = nnx.split(
            vf, nnx.Variable, ...
        )
        self.config = config
        assert config.saveat == diffrax.SaveAt(t1=True), "saveat must be t1=True"

    def vf(self, t, state, args):
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
        config = self.config.optional_override(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            saveat=saveat,
        )
        term = diffrax.ODETerm(self.vf)
        y0 = (x, log_density)
        sol = config.solve(term, y0, (self.vf_variables, kwargs))
        return sol

    def forward(self, x, log_density, **kwargs):
        sol = self.solve_flow(x, log_density, **kwargs)
        return jax.tree.map(lambda x: x[0], sol.ys)

    def reverse(self, x, log_density, **kwargs):
        sol = self.solve_flow(
            x,
            log_density,
            t_start=self.config.t_end,
            t_end=self.config.t_start,
            **kwargs,
        )
        return jax.tree.map(lambda x: x[0], sol.ys)


class ContFlowRK4(Bijection):
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
        t_start = t_start if t_start is not None else self.t_start
        t_end = t_end if t_end is not None else self.t_end
        steps = steps if steps is not None else self.steps

        dt = (t_end - t_start) / steps

        def vf(t, state, args):
            x, log_density = state
            dx_dt, dld_dt = self.vf(t, x, **args)
            return dx_dt, dld_dt

        y0 = (x, log_density)
        y_final = odeint_rk4(
            vf,
            y0,
            t_end,
            kwargs,
            step_size=dt,
            start_time=t_start,
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
    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        # default to single gauge object
        is_lie: tp.Any = True,
        *,
        t_start: float = 0,
        t_end: float = 1,
        steps: int = 20,
        tableau: cg.ButcherTableau = cg.CG3,
    ):
        self.vf = vf
        self.is_lie = is_lie
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
        t_start = t_start if t_start is not None else self.t_start
        t_end = t_end if t_end is not None else self.t_end
        steps = steps if steps is not None else self.steps

        dt = (t_end - t_start) / steps

        def vf(t, state, args):
            x, log_density = state
            dx_dt, dld_dt = self.vf(t, x, **args)
            return dx_dt, dld_dt

        y0 = (x, log_density)
        is_lie = (self.is_lie, False)

        y_final = cg.crouch_grossmann(
            vf,
            y0,
            kwargs,
            t_start,
            t_end,
            step_size=dt,
            is_lie=is_lie,
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

    info = ShapeInfo(event_dim=event_dim)

    def func_flat(x_flat, event_shape):
        x = x_flat.reshape(x_flat.shape[:-1] + event_shape)
        out = func(x)
        return out.reshape(out.shape[:-len(event_shape)] + (-1,))

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, -1))
    def _jvp(x, tang):
        x_flat, _info = info.process_and_flatten(x)
        _func = partial(func_flat, event_shape=_info.event_shape)
        v, jac = jax.jvp(_func, (x_flat,), (tang,))
        return v.reshape(v.shape[:-1] + _info.event_shape), jac

    @auto_vmap(event_dim)
    def call_and_jac(x):
        _, _info = info.process_event(jnp.shape(x))
        tang_basis = jnp.eye(_info.event_size)
        v, jac = _jvp(x, tang_basis)
        return v, jac

    return call_and_jac


class AutoJacVF(nnx.Module):

    def __init__(self, vector_field_base, event_dim=1):
        self.vector_field_base = vector_field_base
        self.event_dim = event_dim

    def __call__(self, t, x):
        jac_fn = _ndim_jacobian(partial(self.vector_field_base, t), self.event_dim)
        v, jac = jac_fn(x)
        return v, -jnp.trace(jac, axis1=-2, axis2=-1)
