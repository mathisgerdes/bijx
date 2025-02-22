"""
Simple neural networks provided for convenience.
"""

import typing as tp

import jax.numpy as jnp
from flax import nnx


class SimpleConvNet(nnx.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
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
                list(hidden_channels) + [out_channels]
            )
        ]

    def __call__(self, x):
        for conv in self.conv_layers[:-1]:
            x = conv(x)
            x = self.activation(x)
        x = self.conv_layers[-1](x)
        x = self.final_activation(x)
        return x


class SimpleResNet(nnx.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            width: int = 1024,
            depth: int = 3,
            *,
            activation: tp.Callable = nnx.gelu,
            final_activation: tp.Callable = lambda x: x,
            dropout: float = 0.0,
            final_bias_init: tp.Callable = nnx.initializers.zeros,
            final_kernel_init: tp.Callable = nnx.initializers.lecun_normal(),
            rngs: nnx.Rngs,
        ):

        # Store configuration
        self.width = width
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.dropout_rate = dropout

        # First layer with special initialization
        self.first_layer = nnx.Linear(in_features=in_features, out_features=width, rngs=rngs)

        # Hidden layers
        self.hidden_layers = [
            nnx.Linear(in_features=width, out_features=width, rngs=rngs)
            for _ in range(depth)
        ]

        # Final layer
        self.final_layer = nnx.Linear(
            in_features=width,
            out_features=out_features,
            kernel_init=final_kernel_init,
            bias_init=final_bias_init,
            rngs=rngs
        )

        # Dropout layer
        self.dropout = nnx.Dropout(rate=dropout)

    def __call__(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            delta = self.dropout(x)
            delta = self.activation(delta)
            delta = layer(delta)
            x += delta
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x
