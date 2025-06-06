"""
Continuous-time vector fields.
"""

import typing as tp

import diffrax
import flax.struct
import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .base import Bijection
from .rk4ode import odeint_rk4


@flax.struct.dataclass
class DiffraxConfig:
    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.05
    saveat: diffrax.SaveAt = diffrax.SaveAt(t1=True)
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
    adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()
    event: diffrax.Event | None = None
    max_steps: int | None = 4096
    throw: bool = True
    solver_state: ftp.ArrayPytree | None = None
    controller_state: ftp.ArrayPytree | None = None
    made_jump: bool | None = None

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


class ODESolver(nnx.Module):
    """Basic wrapper around diffrax.diffeqsolve.

    Example:
        >>> m = lfx.ODESolver(lambda t, y: -y)
        >>> jnp.isclose(m.solve(1.0), jnp.exp(-1.0))
        Array(True, dtype=bool)

    """

    def __init__(
        self,
        vector_field: tp.Callable,
        config: DiffraxConfig = DiffraxConfig(),
        *,
        unpack: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        self.vector_field = vector_field
        self.config = config
        self.unpack = unpack

        assert not unpack or config.saveat == diffrax.SaveAt(
            t1=True
        ), "To automatically unpack the solution, saveat must be t1=True"

    def initialize(self, state, **kwargs):
        def term(t, y, kwargs):
            return self.vector_field(t, y, **kwargs)

        term = diffrax.ODETerm(term)
        # term, y0, args
        return term, state, kwargs

    def solve(
        self,
        state,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        dt: float | None = None,
        saveat: diffrax.SaveAt | None = None,
        solver_state: ftp.ArrayPytree | None = None,
        **kwargs,
    ):
        config = self.config.optional_override(
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            saveat=saveat,
            solver_state=solver_state,
        )

        term, y0, args = self.initialize(state, **kwargs)
        sol = config.solve(term, y0, args)

        if self.unpack:
            return jax.tree.map(lambda x: x[0], sol.ys)
        return sol


class ContFlowDiffrax(Bijection):
    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        config: DiffraxConfig = DiffraxConfig(),
    ):
        self.vf = vf
        self.config = config
        assert config.saveat == diffrax.SaveAt(t1=True), "saveat must be t1=True"

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
        term = diffrax.ODETerm(lambda t, state, args: self.vf(t, state[0], **args))
        y0 = (x, log_density)
        sol = config.solve(term, y0, kwargs)
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
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end
        if steps is None:
            steps = self.steps

        # odeint_rk4 only supports positive step sizes
        dt = np.abs(1 / steps)
        delta_t = np.abs(t_end - t_start)
        negative = t_start > t_end

        def vf(t, state, args):
            if negative:
                dx, dd = self.vf(t_start - t, state[0], **args)
                return -dx, -dd
            else:
                return self.vf(t_start + t, state[0], **args)

        y0 = (x, log_density)
        x, log_density = odeint_rk4(
            vf,
            y0,
            delta_t,
            kwargs,
            step_size=dt,
            start_time=0.0,
        )
        return x, log_density

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
