r"""
Nonlinear feature transformations for neural network layers.

This module provides nonlinear feature mappings for the vector fields of
continuous normalizing flows.

For continuous normalizing flows, the divergence of the feature map is computed
automatically using the vector-Jacobian product:

$$
\nabla \cdot \mathbf{f} =
\text{tr}\left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)
$$

This enables efficient computation of log-density changes in normalizing flows,
as the non-linear features are applied "locally".
"""

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class NonlinearFeatures(nnx.Module):
    """Base class for nonlinear feature transformations with divergence computation.

    Provides the foundation for feature mappings that transform input data through
    learned nonlinear functions. Automatically computes the divergence of the
    transformation using vector-Jacobian products.

    Args:
        out_channel_size: Total number of output feature channels.
        rngs: Random number generator state for parameter initialization.

    Note:
        This is an abstract base class. Subclasses must implement
        :meth:`apply_feature_map` to define the specific nonlinear transformation.
        The divergence computation is handled automatically by the base class.
    """

    def __init__(
        self,
        out_channel_size: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.out_channel_size = out_channel_size

    def apply_feature_map(self, inputs, **kwargs):
        """Apply the nonlinear feature transformation.

        This method must be implemented by subclasses to define the specific
        nonlinear mapping applied to input data.

        Args:
            inputs: Input data to transform.
            **kwargs: Additional transformation-specific arguments.

        Returns:
            Transformed feature representation.
        """
        raise NotImplementedError()

    def __call__(
        self, inputs, local_coupling, flatten_features=True, mask=None, **kwargs
    ):
        """Apply feature transformation with automatic divergence computation.

        Computes both the nonlinear feature transformation and its divergence
        using the provided local coupling matrix. The divergence is computed
        using vector-Jacobian products to avoid explicit Jacobian
        matrix construction.

        Args:
            inputs: Input data with shape (..., spatial_dims, channels).
            local_coupling: Local coupling matrix from convolution kernel diagonal.
            flatten_features: Whether to flatten feature dimensions.
            mask: Optional mask for selective divergence computation.
            **kwargs: Additional arguments passed to feature transformation.

        Returns:
            Tuple of (transformed_features, divergence) where:
                - transformed_features: Nonlinearly transformed input features
                - divergence: Divergence of the transformation for density tracking

        Note:
            The local_coupling matrix typically comes from the diagonal elements
            of a convolution kernel, representing site-wise coupling strengths.
            The divergence computation uses the vector-Jacobian product for efficiency.
        """
        # Compute divergence using local couplings (W_xx part of conv kernel)
        # Feature map is exclusively site-wise
        orig_channels = inputs.shape[-1]
        apply = partial(self.apply_feature_map, **kwargs)
        inputs, bwd = jax.vjp(apply, inputs)

        if flatten_features:
            local_coupling = local_coupling.reshape(orig_channels, orig_channels, -1)

        idc = np.arange(local_coupling.shape[1])
        cotangent_reshape = (*inputs.shape[:-2], 1, 1)
        cotangent = jnp.tile(local_coupling[idc, idc], cotangent_reshape)

        if mask is not None:
            (inputs_grad,) = bwd(cotangent * jnp.expand_dims(mask, (-1, -2)))
        else:
            (inputs_grad,) = bwd(cotangent)
        divergence = jnp.sum(inputs_grad, np.arange(1, inputs_grad.ndim))

        if flatten_features:
            inputs = inputs.reshape(inputs.shape[:-2] + (-1,))
        return inputs, divergence


class FourierFeatures(NonlinearFeatures):
    r"""Sinusoidal Fourier feature transformation with learnable frequencies.

    The frequencies $\mathbf{\omega}_i$ are learned parameters initialized from
    a uniform distribution, allowing the network to adapt to the characteristic
    scales present in the data.

    Args:
        feature_count: Number of sinusoidal features per input channel.
        input_channels: Number of input channels to transform.
        freq_init: Initializer for frequency parameters.
        rngs: Random number generator state.

    Note:
        The total output size is input_channels × feature_count.

    Example:
        >>> features = FourierFeatures(16, input_channels=1, rngs=rngs)
        >>> transformed, div = features(phi[..., None], local_coupling)
    """

    def __init__(
        self,
        feature_count: int,
        input_channels: int,
        *,
        freq_init: tp.Callable = nnx.initializers.uniform(5.0),
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(input_channels * feature_count, rngs=rngs)
        self.feature_count = feature_count

        self.phi_freq = nnx.Param(
            freq_init(rngs.params(), (input_channels, feature_count))
        )

    def apply_feature_map(self, phi_lin, **kwargs):
        """Apply sinusoidal feature transformation.

        Args:
            phi_lin: Input data to transform.
            **kwargs: Additional arguments (unused).

        Returns:
            Sinusoidal features with shape (..., input_channels, feature_count).
        """
        features = jnp.einsum("...i,ij->...ij", phi_lin, self.phi_freq.value)
        features = jnp.sin(features)
        return features


class PolynomialFeatures(NonlinearFeatures):
    r"""Polynomial feature transformation with specified powers.

    Transforms input data through polynomial basis functions of specified
    degrees.

    The transformation applies each specified power element-wise to the input,
    creating a polynomial basis that can represent complex nonlinear relationships.

    Args:
        powers: List of polynomial powers to apply.
        input_channels: Number of input channels to transform.
        rngs: Random number generator state.

    Note:
        Powers should be non-negative integers. The power 0 gives constant
        features (all ones), power 1 gives identity, and higher powers
        provide increasingly nonlinear transformations.

    Example:
        >>> # Polynomial features with linear and quadratic terms
        >>> features = PolynomialFeatures([1, 2], input_channels=1, rngs=rngs)
        >>> transformed, div = features(jnp.ones((1, 1)), jnp.ones((1, 2)))

    Important:
        Inclusion of powers other than 0 and 1 can lead to numerical instability
        as the vector fields may not be Lipschitz continuous.
    """

    def __init__(
        self,
        powers: list[int],
        input_channels: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(input_channels * len(powers), rngs=rngs)
        self.powers = powers

    def apply_feature_map(self, phi_lin, **kwargs):
        """Apply polynomial feature transformation.

        Args:
            phi_lin: Input data to transform.
            **kwargs: Additional arguments (unused).

        Returns:
            Polynomial features with shape (..., input_channels, len(powers)).
        """
        features = jnp.stack([phi_lin**p for p in self.powers], axis=-1)
        return features


class ConcatFeatures(NonlinearFeatures):
    """Concatenation of multiple feature maps.

    Combines multiple nonlinear feature maps by applying each
    transformation to the input and concatenating the results.

    Args:
        features: List of NonlinearFeatures instances to compose.
        rngs: Random number generator state.

    Note:
        The total output size is the sum of all component feature sizes.
        This approach allows combining complementary feature types (e.g.,
        Fourier and polynomial features) for higher expressiveness.

    Example:
        >>> fourier = FourierFeatures(49, input_channels=1, rngs=rngs)
        >>> poly = PolynomialFeatures([1, 2], input_channels=1, rngs=rngs)
        >>> combined = ConcatFeatures([fourier, poly], rngs=rngs)
        >>> combined.out_channel_size == 49 + 2
        True
    """

    def __init__(
        self,
        features: nnx.List[NonlinearFeatures],
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(sum(f.out_channel_size for f in features), rngs=rngs)
        self.features = nnx.List(features)

    def apply_feature_map(self, phi_lin, **kwargs):
        """Apply all component feature transformations and concatenate results.

        Args:
            phi_lin: Input data to transform.
            **kwargs: Additional arguments passed to all component transformations.

        Returns:
            Concatenated features from all component transformations.
        """
        return jnp.concatenate(
            [f.apply_feature_map(phi_lin, **kwargs) for f in self.features], axis=-1
        )
