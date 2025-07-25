"""
Continuous-time vector fields.
"""

import typing as tp

import diffrax
import flax.nnx as nnx
import jax

from ..solvers import DiffraxConfig, odeint_rk4
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
