r"""
Convolutional continuous normalizing flows with symmetry preservation.

This module implements convolutional continuous normalizing flows (Conv-CNFs)
that preserve spatial symmetries while learning complex transformations.
These flows are particularly suited for scalar lattice field theory,
and other structured data with high degree of spatial symmetry.

Mathematical background:
Conv-CNFs use vector fields of the form
$$\frac{d\mathbf{x}}{dt} = \text{Conv}(f(\mathbf{x}), K(t))$$
where $K(t)$ is a time-dependent convolution kernel and $f$ is a feature map.
Because the feature map $f$ acts locally, the divergence is tractable.
"""

import typing as tp
from functools import partial

import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from ..nn import embeddings
from ..nn.conv import ConvSym, kernel_d4
from ..nn.features import (
    ConcatFeatures,
    FourierFeatures,
    NonlinearFeatures,
    PolynomialFeatures,
)
from ..utils import ShapeInfo


def _contract_with_emb(par, t_emb):
    """Contract parameter tensor with time embedding."""
    if par.value is None:
        return None
    return rearrange(par.value, "... (c t) -> ... c t", t=t_emb.shape[-1]) @ t_emb


class ConvVF(nnx.Module):
    r"""Convolutional continuous normalizing flow with symmetry preservation.

    Implements a vector field for continuous normalizing flows that uses
    symmetric convolutions to preserve spatial structure. The convolution
    kernels are time-dependent and coupled with nonlinear feature transformations.

    The vector field preserves discrete symmetries (typically D4 group)
    while allowing for complex, spatially-structured transformations.

    Args:
        shape_info: Shape information for spatial and channel dimensions.
        conv: Symmetric convolution layer with time-dependent parameters.
        time_kernel: Time embedding module for kernel modulation.
        feature_map: Nonlinear feature transformation applied before convolution.
        feature_superposition: Optional feature dimensionality reduction.

    Note:
        Most conveniently constructed using :meth:`ConvVF.build`.

    Example:
        >>> # Build conv CNF for 2D lattice
        >>> cnf = ConvVF.build(
        ...     kernel_shape=(3, 3),
        ...     channel_shape=(),
        ...     features=(
        ...         partial(FourierFeatures, 49),
        ...         partial(PolynomialFeatures, (1,)),
        ...     ),
        ...     rngs=rngs
        ... )
        >>> velocity, div = cnf(t, phi)
    """

    def __init__(
        self,
        *,
        shape_info: ShapeInfo,
        conv: ConvSym,
        time_kernel: nnx.Module,
        feature_map: NonlinearFeatures,
        feature_superposition: nnx.Variable | None = None,
    ):
        self.shape_info = shape_info
        self.conv = conv
        self.time_kernel = time_kernel
        self.feature_map = feature_map
        self.feature_superposition = feature_superposition

    def __call__(self, t, x):
        """Compute vector field and divergence for Conv-CNF.

        Args:
            t: Time parameter.
            x: Input spatial array with shape (..., height, width, channels).

        Returns:
            Tuple of (velocity_field, negative_divergence) where:
            - velocity_field: Spatial velocity field dx/dt
            - negative_divergence: Negative divergence for log-density evolution
        """
        batch_shape, shape_info = self.shape_info.process_event(x.shape)
        channel_size = shape_info.channel_size
        x = x.reshape(-1, *shape_info.space_shape, channel_size)

        t_emb = self.time_kernel(t)

        # contract time embedding with conv kernel & bias
        conv_graph, conv_params = nnx.split(self.conv)

        conv_params["kernel_params"] = nnx.Param(
            _contract_with_emb(conv_params["kernel_params"], t_emb)
        )
        if self.conv.bias is not None:
            conv_params["bias"] = nnx.Param(
                _contract_with_emb(conv_params["bias"], t_emb)
            )

        conv = nnx.merge(conv_graph, conv_params)

        feature_superposition = (
            self.feature_superposition.value / self.feature_map.out_channel_size
        )

        # extract the local-coupling weights; shape=(in features, out features)
        w00 = conv.kernel_params.value[0]
        # contract with feature superposition
        w00 = jnp.einsum("if,io->fo", feature_superposition, w00)

        features, div = self.feature_map(x, w00)
        features = jnp.einsum("fw,...w->...f", feature_superposition, features)
        grad_phi = conv(features)

        grad_phi = grad_phi.reshape(*batch_shape, *shape_info.event_shape)
        return grad_phi, -div.reshape(*batch_shape)

    @classmethod
    def build(
        cls,
        kernel_shape,
        channel_shape: tuple[int, ...] = (),
        *,
        symmetry: tp.Callable = kernel_d4,
        use_bias: bool = False,
        time_kernel: nnx.Module = embeddings.KernelFourier(21),
        time_kernel_reduced=20,
        features: tuple[NonlinearFeatures, ...] = (
            partial(FourierFeatures, 49),
            partial(PolynomialFeatures, (1,)),
        ),
        features_reduced: int | None = 20,
        rngs: nnx.Rngs,
    ):
        """Build a ConvVF with default architecture choices.

        Constructs a complete convolutional CNF by assembling symmetric convolutions,
        time embeddings, and feature transformations with sensible defaults.
        In particular, it takes care of enlarging internal parameters with
        an extra axis, which is later contracted with the time embedding.

        Args:
            kernel_shape: Spatial shape of convolution kernels (e.g., (3, 3)).
            channel_shape: Shape of channel dimensions, defaults to scalar channels.
            symmetry: Symmetry group operation (default D4 rotations and reflections).
            use_bias: Whether to include bias terms in convolutions.
            time_kernel: Time embedding module for kernel modulation.
            time_kernel_reduced: Dimensionality reduction for time embeddings.
            features: Tuple of feature transformation classes to compose.
            features_reduced: Dimensionality reduction for feature superposition.
            rngs: Random number generator state for parameter initialization.

        Returns:
            ConvVF instance to be used in continuous normalizing flows.

        Example:
            >>> # Standard 2D lattice CNF with Fourier + linear features
            >>> cnf = ConvVF.build(
            ...     kernel_shape=(3, 3),
            ...     channel_shape=(1,),  # scalar field
            ...     features_reduced=16,
            ...     rngs=rngs
            ... )
        """
        channel_size = np.prod(channel_shape, dtype=int)

        if time_kernel_reduced is not None:
            time_kernel = embeddings.KernelReduced(
                time_kernel, time_kernel_reduced, rngs=rngs
            )

        features = ConcatFeatures(
            [f_map(channel_size, rngs=rngs) for f_map in features]
        )

        conv_in_features = features.out_channel_size
        if features_reduced is not None:
            feature_superposition = nnx.Param(
                nnx.initializers.orthogonal()(
                    rngs.params(), (features_reduced, features.out_channel_size)
                )
            )
            conv_in_features = features_reduced

        conv = ConvSym(
            in_features=conv_in_features,
            out_features=channel_size * time_kernel.feature_count,
            kernel_size=kernel_shape,
            orbit_function=symmetry,
            rngs=rngs,
            use_bias=use_bias,
        )

        shape_info = ShapeInfo(
            space_dim=len(kernel_shape), channel_dim=len(channel_shape)
        )

        return cls(
            shape_info=shape_info,
            conv=conv,
            time_kernel=time_kernel,
            feature_map=features,
            feature_superposition=feature_superposition,
        )
