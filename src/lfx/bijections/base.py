from functools import partial

import jax
from flax import nnx

from ..utils import Const


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
    def __init__(self, bijection: Bijection, invert: bool = True):
        self.invert = Const(invert)
        self.bijection = bijection

    def forward(self, x, log_density, **kwargs):
        return jax.lax.cond(
            self.invert.value,
            lambda x, ld, kw: self.bijection.reverse(x, ld, **kw),
            lambda x, ld, kw: self.bijection.forward(x, ld, **kw),
            x,
            log_density,
            kwargs,
        )

    def reverse(self, x, log_density, **kwargs):
        return jax.lax.cond(
            self.invert.value,
            lambda x, ld, kw: self.bijection.forward(x, ld, **kw),
            lambda x, ld, kw: self.bijection.reverse(x, ld, **kw),
            x,
            log_density,
            kwargs,
        )


class Chain(Bijection):
    def __init__(self, *bijections: Bijection):
        self.bijections = list(bijections)

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


class ScanChain(Bijection):

    def __init__(self, stack):
        self.stack = stack

    def _forward(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.forward(*carry, **kwargs), None

    def _reverse(self, carry, variables, graph, **kwargs):
        bijection = nnx.merge(graph, variables)
        return bijection.reverse(*carry, **kwargs), None

    def forward(self, x, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (y, lp), _ = jax.lax.scan(
            partial(self._forward, graph=graph, **kwargs),
            (x, log_density),
            variables,
        )
        return y, lp

    def reverse(self, y, log_density, **kwargs):
        graph, variables = nnx.split(self.stack)
        (x, lp), _ = jax.lax.scan(
            partial(self._reverse, graph=graph, **kwargs),
            (y, log_density),
            variables,
            reverse=True,
        )
        return x, lp


class Frozen(Bijection):
    def __init__(self, bijection: Bijection):
        self.frozen = bijection

    def forward(self, x, log_density, **kwargs):
        return self.frozen.forward(x, log_density, **kwargs)

    def reverse(self, x, log_density, **kwargs):
        return self.frozen.reverse(x, log_density, **kwargs)
        return self.frozen.reverse(x, log_density, **kwargs)
