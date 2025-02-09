import typing as tp

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class Const(nnx.Variable): pass

class Bijection(nnx.Module):
    def forward(self, x, log_density, **kwargs):
        return x, log_density

    def reverse(self, x, log_density, **kwargs):
        return x, log_density

    def __call__(self, x, log_density, **kwargs):
        return self.forward(x, log_density, **kwargs)


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

    def forward(self, x, log_density, *, arg_list: list[dict]|None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(self.bijections, arg_list, strict=True):
            x, log_density = bijection.forward(x, log_density, *args, **kwargs)
        return x, log_density

    def reverse(self, x, log_density, *, arg_list: list[dict]|None = None, **kwargs):
        if arg_list is None:
            arg_list = [{}] * len(self.bijections)
        for bijection, args in zip(reversed(self.bijections), reversed(arg_list), strict=True):
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
        if isinstance(shape_or_val, (jax.Array, np.ndarray, int, float)):
            self.scale_val = Const(shape_or_val)
        else:
            if rngs is None:
                raise ValueError("rngs must be provided if shape_or_val is not a constant")
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
        if isinstance(shape_or_val, (jax.Array, np.ndarray, int, float)):
            self.shift_val = Const(shape_or_val)
        else:
            if rngs is None:
                raise ValueError("rngs must be provided if shape_or_val is not a constant")
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


filter_frozen = nnx.Any(Const, nnx.PathContains('frozen'))
