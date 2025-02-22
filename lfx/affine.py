import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .bijections import Bijection, Const


def checker_mask(shape, parity: bool):
    """Checkerboard mask.

    Args:
        shape: Spacial dimensions of input.
        parity: Parity of mask.
    """
    idx_shape = np.ones_like(shape)
    idc = []
    for i, s in enumerate(shape):
        idx_shape[i] = s
        idc.append(np.arange(s, dtype=np.uint8).reshape(idx_shape))
        idx_shape[i] = 1
    mask = (sum(idc) + parity) % 2
    return mask


class SimpleConvNet(nnx.Module):

    def __init__(
            self,
            in_channels: int,
            kernel_size: tuple[int, ...] = (3, 3),
            hidden_channels: list[int] = [8, 8],
            activation: tp.Callable = nnx.leaky_relu,
            final_activation: tp.Callable = nnx.tanh,
            *,
            rngs: nnx.Rngs,
        ):
        self.kernel_size = kernel_size
        self.activation = activation
        self.final_activation = final_activation

        self.conv_layers = [
            nnx.Conv(
                in_features=c_in,
                out_features=c_out,
                kernel_size=kernel_size,
                padding='CIRCULAR',
                rngs=rngs,
            )
            for c_in, c_out in zip(
                [in_channels] + list(hidden_channels),
                list(hidden_channels) + [in_channels * 2]
            )
        ]

    def __call__(self, x):
        for conv in self.conv_layers[:-1]:
            x = conv(x)
            x = self.activation(x)
        x = self.conv_layers[-1](x)
        x = self.final_activation(x)
        return x


class AffineCoupling(Bijection):
    """
    Affine coupling layer.

    Example:
    ```python
    space_shape = (16, 16)

    affine_flow = lfx.Chain([
        lfx.ExpandDims(),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), True)),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), False)),
        lfx.ExpandDims().invert(),
    ])
    ```

    Args:
        net: Network that maps frozen features to s, t.
        mask: Mask to apply to input.
    """

    def __init__(self, net: nnx.Module, mask: jax.Array, *, rngs=None):
        self.mask = Const(mask)
        # x_f -> act such that s,t=split(act, 2, -1) and
        # x_out = t + x_a * exp(s) + x_f
        self.net = net

    @property
    def mask_active(self):
        return 1 - self.mask.value

    @property
    def mask_frozen(self):
        return self.mask.value

    def forward(self, x, log_density):
        x_frozen = self.mask_frozen * x
        x_active = self.mask_active * x
        activation = self.net(x_frozen)
        s, t = jnp.split(activation, 2, -1)
        fx = x_frozen + self.mask_active * t + x_active * jnp.exp(s)
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return fx, log_density - log_jac

    def reverse(self, fx, log_density):
        fx_frozen = self.mask_frozen * fx
        fx_active = self.mask_active * fx
        activation = self.net(fx_frozen)
        s, t = jnp.split(activation, 2, -1)
        x = (fx_active - self.mask_active * t) * jnp.exp(-s) + fx_frozen
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return x, log_density + log_jac
