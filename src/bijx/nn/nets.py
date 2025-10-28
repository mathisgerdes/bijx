r"""
Complete neural network architectures for convenience and prototyping.

This module provides some common and ready-to-use neural network architectures.
"""

import typing as tp

from flax import nnx


class ConvNet(nnx.Module):
    """Feedforward convolutional neural network.

    Implements a standard convolutional architecture with multiple layers.
    Uses periodic padding by default to respect boundary conditions.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size: Spatial size of convolution kernels.
        hidden_channels: Number of channels in each hidden layer.
        activation: Activation function for hidden layers.
        final_activation: Activation function for output layer.
        padding: Padding mode ('CIRCULAR' for periodic boundaries).
        rngs: Random number generator state.

    Example:
        >>> net = ConvNet(
        ...     in_channels=1, out_channels=2,
        ...     hidden_channels=[16, 32, 16],
        ...     padding="CIRCULAR", rngs=rngs
        ... )
        >>> output = net(lattice_data)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] = (3, 3),
        hidden_channels: list[int] = [8, 8],
        activation: tp.Callable = nnx.leaky_relu,
        final_activation: tp.Callable = nnx.tanh,
        final_kernel_init: tp.Callable = nnx.initializers.normal(),
        final_bias_init: tp.Callable = nnx.initializers.zeros,
        padding: str = "CIRCULAR",
        *,
        rngs: nnx.Rngs,
    ):
        self.kernel_size = kernel_size
        self.activation = activation
        self.final_activation = final_activation

        self.conv_layers = nnx.List(
            [
                nnx.Conv(
                    in_features=c_in,
                    out_features=c_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    rngs=rngs,
                )
                for c_in, c_out in zip(
                    [in_channels] + list(hidden_channels)[:-1],
                    list(hidden_channels),
                )
            ]
        )

        self.conv_layers.append(
            nnx.Conv(
                in_features=hidden_channels[-1],
                out_features=out_channels,
                kernel_size=kernel_size,
                kernel_init=final_kernel_init,
                bias_init=final_bias_init,
                padding=padding,
                rngs=rngs,
            )
        )

    def __call__(self, x):
        """Apply convolutional network to input data.

        Args:
            x: Input tensor with shape (..., height, width, channels).

        Returns:
            Network output.
        """
        for conv in self.conv_layers[:-1]:
            x = conv(x)
            x = self.activation(x)
        x = self.conv_layers[-1](x)
        x = self.final_activation(x)
        return x


class ResNet(nnx.Module):
    """Residual neural network with skip connections.

    Args:
        in_features: Input feature dimensionality.
        out_features: Output feature dimensionality.
        width: Hidden layer width (number of neurons).
        depth: Number of residual blocks.
        activation: Activation function for hidden layers.
        final_activation: Activation function for output layer.
        dropout: Dropout rate for regularization.
        final_bias_init: Initialization for final layer bias.
        final_kernel_init: Initialization for final layer weights.
        rngs: Random number generator state.

    Note:
        The residual connections are applied to the intermediate representations
        of fixed width. The final layer maps to the desired output dimensionality.
        Dropout is applied before each residual block for regularization.

    Example:
        >>> net = ResNet(
        ...     in_features=64, out_features=32,
        ...     width=512, depth=10, rngs=rngs
        ... )
        >>> output = net(features)
    """

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
        self.width = width
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.dropout_rate = dropout

        self.first_layer = nnx.Linear(
            in_features=in_features, out_features=width, rngs=rngs
        )

        self.hidden_layers = nnx.List(
            [
                nnx.Linear(in_features=width, out_features=width, rngs=rngs)
                for _ in range(depth)
            ]
        )

        self.final_layer = nnx.Linear(
            in_features=width,
            out_features=out_features,
            kernel_init=final_kernel_init,
            bias_init=final_bias_init,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(rate=dropout)

    def __call__(self, x):
        """Apply residual network to input features.

        Args:
            x: Input tensor with shape (..., in_features).

        Returns:
            Output tensor with shape (..., out_features).
        """
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            delta = self.dropout(x)
            delta = self.activation(delta)
            delta = layer(delta)
            x += delta
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x


class MLP(nnx.Module):
    """Multi-layer perceptron for general function approximation.

    Implements a standard feedforward neural network with customizable
    architecture and activation functions.

    Args:
        in_features: Input feature dimensionality.
        out_features: Output feature dimensionality.
        hidden_features: List of hidden layer widths.
        activation: Activation function for hidden layers.
        final_activation: Activation function for output layer.
        rngs: Random number generator state.

    Example:
        >>> # MLP for coupling layer transformation
        >>> net = MLP(
        ...     in_features=32, out_features=64,
        ...     hidden_features=[128, 256, 128],
        ...     activation=nnx.gelu, rngs=rngs
        ... )
        >>> output = net(input_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int] = [1024, 1024],
        *,
        activation: tp.Callable = nnx.gelu,
        final_activation: tp.Callable = lambda x: x,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.final_activation = final_activation

        layers = [
            nnx.Linear(
                in_features=in_features, out_features=hidden_features[0], rngs=rngs
            )
        ]

        for i in range(len(hidden_features) - 1):
            layers.append(
                nnx.Linear(
                    in_features=hidden_features[i],
                    out_features=hidden_features[i + 1],
                    rngs=rngs,
                )
            )
        layers.append(
            nnx.Linear(
                in_features=hidden_features[-1], out_features=out_features, rngs=rngs
            )
        )

        self.layers = nnx.List(layers)

    def __call__(self, x):
        """Apply multi-layer perceptron to input features.

        Args:
            x: Input tensor with shape (..., in_features).

        Returns:
            Output tensor with shape (..., out_features).
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
