import typing as tp
from functools import partial

import diffrax
import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .rk4ode import odeint_rk4


class Const(nnx.Variable):
    pass


# filter constants (above) and things wrapped in Frozen (defined below)
filter_frozen = nnx.Any(Const, nnx.PathContains("frozen"))


class Bijection(nnx.Module):
    def forward(self, x, log_density, **kwargs):
        return x, log_density

    def reverse(self, x, log_density, **kwargs):
        return x, log_density

    def __call__(self, x, log_density, **kwargs):
        return self.forward(x, log_density, **kwargs)

    def invert(self):
        return Inverse(self)


class Inverse(Bijection):
    def __init__(self, bijection: Bijection):
        self.bijection = bijection

    def forward(self, x, log_density, **kwargs):
        return self.bijection.reverse(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.bijection.forward(x, log_density, **kwargs)


class Chain(Bijection):
    def __init__(self, bijections: list[Bijection]):
        self.bijections = bijections

    def forward(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(self.bijections, arg_list, strict=True):
            x, log_density = bijection.forward(x, log_density, *args, **kwargs)
        return x, log_density

    def reverse(self, x, log_density, *, arg_list: list[dict] | None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(
            reversed(self.bijections), reversed(arg_list), strict=True
        ):
            x, log_density = bijection.reverse(x, log_density, *args, **kwargs)
        return x, log_density


class Frozen(Bijection):
    def __init__(self, bijection: Bijection):
        self.frozen = bijection

    def forward(self, x, log_density, **kwargs):
        return self.frozen.forward(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.frozen.reverse(x, log_density, **kwargs)


class Scaling(Bijection):
    def __init__(
        self,
        shape_or_val: jax.Array | tuple[int, ...],
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.ones,
        dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        if isinstance(shape_or_val, jax.Array | np.ndarray | int | float):
            self.scale_val = Const(shape_or_val)
        else:
            if rngs is None:
                raise ValueError(
                    "rngs must be provided if shape_or_val is not a constant"
                )
            self.scale_val = nnx.Param(init(rngs.params(), shape_or_val, dtype))

    @property
    def scale(self):
        try:
            scale = self.scale_val.value
        except AttributeError:
            scale = self.scale_val
        return self.transform(scale)

    def forward(self, x, log_density, **kwargs):
        return x * self.scale, log_density - jnp.sum(jnp.log(jnp.abs(self.scale)))

    def reverse(self, x, log_density, **kwargs):
        return x / self.scale, log_density + jnp.sum(jnp.log(jnp.abs(self.scale)))


class Shift(Bijection):
    def __init__(
        self,
        shape_or_val: jax.Array | tuple[int, ...],
        transform: tp.Callable = lambda x: x,
        *,
        init: ftp.Initializer = nnx.initializers.zeros,
        dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.transform = transform
        if isinstance(shape_or_val, jax.Array | np.ndarray | int | float):
            self.shift_val = Const(shape_or_val)
        else:
            if rngs is None:
                raise ValueError(
                    "rngs must be provided if shape_or_val is not a constant"
                )
            self.shift_val = nnx.Param(init(rngs.params(), shape_or_val, dtype))

    @property
    def shift(self):
        try:
            shift = self.shift_val.value
        except AttributeError:
            shift = self.shift_val
        return self.transform(shift)

    def forward(self, x, log_density, **kwargs):
        return x + self.shift, log_density

    def reverse(self, x, log_density, **kwargs):
        return x - self.shift, log_density


class MetaLayer(Bijection):
    """Convenience class for operations that do not change density."""

    def __init__(self, forward: tp.Callable, reverse: tp.Callable, *, rngs=None):
        self._forward = forward
        self._reverse = reverse

    def forward(self, x, log_density):
        return self._forward(x), log_density

    def reverse(self, x, log_density):
        return self._reverse(x), log_density


class ExpandDims(MetaLayer):
    def __init__(self, axis: int = -1, *, rngs=None):
        super().__init__(
            partial(jnp.expand_dims, axis=axis),
            partial(jnp.squeeze, axis=axis),
        )


class ContFlowDiffrax(Bijection):
    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        *,
        t_start: float = 0,
        t_end: float = 1,
        dt: float = 1 / 20,
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize(),  # noqa: E501
        # switch to diffrax.BacksolveAdjoint() "adjoint sensitivity"; lower memory usage
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
    ):
        self.vf = vf
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint

    def solve_flow(
        self,
        x,
        log_density,
        *,
        # integration parameters
        t_start=None,
        t_end=None,
        dt=None,
        saveat: diffrax.SaveAt | None = diffrax.SaveAt(t1=True),
        max_steps: int | None = 4096,
        # arguments to vector field
        **kwargs,
    ):
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end
        if dt is None:
            dt = self.dt

        term = diffrax.ODETerm(lambda t, state, args: self.vf(t, state[0], **args))
        y0 = (x, log_density)
        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t_start,
            t1=t_end,
            dt0=dt,
            y0=y0,
            args=kwargs,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=max_steps,
        )
        return sol

    def forward(self, x, log_density, **kwargs):
        sol = self.solve_flow(x, log_density, **kwargs)
        # TODO: might want to tree-map indexing x[0] instead of (x,) for the
        # general case that x is not an array
        (x,), (log_density,) = sol.ys
        return x, log_density

    def reverse(self, x, log_density, **kwargs):
        sol = self.solve_flow(
            x,
            log_density,
            t_start=self.t_end,
            t_end=self.t_start,
            dt=-self.dt,
            **kwargs,
        )
        (x,), (log_density,) = sol.ys
        return x, log_density


class ContFlowRK4(Bijection):
    def __init__(
        self,
        # (t, x, **kwargs) -> dx/dt, d(log_density)/dt
        vf: tp.Callable,
        *,
        t_start: float = 0,
        t_end: float = 1,
        dt: float = 1 / 20,
    ):
        self.vf = vf
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt

    def solve_flow(
        self,
        x,
        log_density,
        *,
        # integration parameters
        t_start=None,
        t_end=None,
        dt=None,
        # arguments to vector field
        **kwargs,
    ):
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end
        if dt is None:
            dt = self.dt

        # odeint_rk4 only supports positive step sizes
        dt = np.abs(dt)
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
            dt=-self.dt,
            **kwargs,
        )
