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

    def __call__(self, inputs, **kwargs):
        """Return ``(features, div_map)`` for the given inputs.

        Args:
            inputs: Input array with shape ``(..., channels)``.
            **kwargs: Passed through to :meth:`apply_feature_map`.

        Returns:
            features: Flattened feature array ``(..., channels * feature_count)``.
            div_map: Callable ``(local_coupling, mask=None) -> divergence`` where
                ``local_coupling`` has shape ``(..., F_total, O)`` (site-dependent)
                or ``(F_total, O)`` (stationary), with ``F_total = channels *
                feature_count`` and ``O`` the number of output channels to sum over.
        """
        apply = partial(self.apply_feature_map, **kwargs)
        outputs, bwd = jax.vjp(apply, inputs)
        outputs_shape = outputs.shape  # (..., C, F) — captured before rebind

        def div_map(local_coupling, mask=None):
            # local_coupling: (*spatial, F_total, O) or (F_total, O)
            lc = local_coupling.sum(-1)  # (*spatial, F_total) or (F_total,)
            if lc.ndim == 1:
                cotangent = jnp.broadcast_to(
                    lc.reshape(outputs_shape[-2:]), outputs_shape
                )
            else:
                cotangent = lc.reshape(outputs_shape)
            if mask is not None:
                cotangent = cotangent * jnp.expand_dims(mask, (-1, -2))
            (inputs_grad,) = bwd(cotangent)
            return jnp.sum(inputs_grad, np.arange(1, inputs_grad.ndim))

        outputs = outputs.reshape(outputs.shape[:-2] + (-1,))
        return outputs, div_map


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
        The total output size is input_channels * feature_count.

    Example:
        >>> features = FourierFeatures(16, input_channels=1, rngs=rngs)
        >>> transformed, div_map = features(phi[..., None])
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
        features = jnp.einsum("...i,ij->...ij", phi_lin, self.phi_freq)
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
        >>> transformed, div_map = features(jnp.ones((1, 1)))

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


class DecayingFourierFeatures(NonlinearFeatures):
    r"""Sinusoidal features damped by a learnable Gaussian envelope.

    Each feature is
    $$g_k(\phi) = \sin(\omega_k \phi) \, \exp\!\left(-\frac{\phi^2}{2 \ell^2}\right),$$
    where $\ell > 0$ is a learnable (per-input-channel) decay length.

    Matches :class:`FourierFeatures` near $\phi=0$ ($g_k'(0) = \omega_k$ identically),
    but suppresses the field at large $|\phi|$ — killing the periodic wrap-around
    that bare ``sin`` features inherit. Odd in $\phi$, so a linear map without bias
    yields a $\mathbb{Z}_2$-equivariant vector field, just like ``FourierFeatures``.

    With ``log_decay_length_init`` large (default 3, so $\ell \approx 20$), the envelope
    is essentially flat across typical data ranges and the features start out
    indistinguishable from :class:`FourierFeatures`. Training can then shrink
    $\ell$ if a tighter envelope helps.

    Args:
        feature_count: Number of sinusoidal features per input channel.
        input_channels: Number of input channels to transform.
        freq_init: Initializer for frequency parameters.
        log_decay_length_init: Initial value of $\log \ell$ (per input channel).
        rngs: Random number generator state.
    """

    def __init__(
        self,
        feature_count: int,
        input_channels: int,
        *,
        freq_init: tp.Callable = nnx.initializers.uniform(5.0),
        log_decay_length_init: float = 3.0,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(input_channels * feature_count, rngs=rngs)
        self.feature_count = feature_count

        self.phi_freq = nnx.Param(
            freq_init(rngs.params(), (input_channels, feature_count))
        )
        self.log_decay_length = nnx.Param(
            jnp.full((input_channels,), log_decay_length_init, dtype=jnp.float32)
        )

    def apply_feature_map(self, phi_lin, **kwargs):
        decay_length = jnp.exp(self.log_decay_length)
        # phi_lin: (..., C); envelope per-channel
        envelope = jnp.exp(-0.5 * (phi_lin / decay_length) ** 2)  # (..., C)
        sinusoid = jnp.sin(jnp.einsum("...i,ij->...ij", phi_lin, self.phi_freq))
        return sinusoid * envelope[..., None]


class GaussianRBFFeatures(NonlinearFeatures):
    r"""Gaussian radial basis features with learnable centers and widths.

    Each feature is the antisymmetrized RBF
    $$g_k(\phi) = \exp\!\left(-\frac{(\phi - c_k)^2}{2 \sigma_k^2}\right)
                 - \exp\!\left(-\frac{(\phi + c_k)^2}{2 \sigma_k^2}\right),$$
    with $c_k > 0$ (parameterized via softplus so it stays positive under training).
    The result is **odd in $\phi$** by construction, so a linear map without bias
    produces a $\mathbb{Z}_2$-equivariant vector field — matching the inductive
    bias that $\sin$ features have in ConvVF, while also being non-periodic,
    smooth, and decaying outside the data range.

    Args:
        feature_count: Number of RBFs per input channel.
        input_channels: Number of input channels.
        center_max: Centers $c_k$ are initialized evenly on $(0, c_{\max}]$.
            Default ``3.0`` covers typical $\phi^4$ field values.
        log_sigma_init: Initial value of $\log \sigma$. Default ``log(0.8)``.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        feature_count: int,
        input_channels: int,
        *,
        center_max: float = 3.0,
        log_sigma_init: float = float(np.log(0.8)),
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(input_channels * feature_count, rngs=rngs)
        self.feature_count = feature_count

        # init centers evenly on (0, center_max]; store the pre-softplus value
        centers_init = jnp.linspace(
            center_max / feature_count, center_max, feature_count
        )
        # invert softplus: c = log(exp(c_pos) - 1)
        raw_init = jnp.log(jnp.expm1(centers_init))
        raw_init = jnp.broadcast_to(raw_init, (input_channels, feature_count))
        self._raw_centers = nnx.Param(jnp.asarray(raw_init, dtype=jnp.float32))
        self.log_sigma = nnx.Param(
            jnp.full((input_channels, feature_count), log_sigma_init, dtype=jnp.float32)
        )

    @property
    def centers(self):
        return jax.nn.softplus(self._raw_centers)

    def apply_feature_map(self, phi_lin, **kwargs):
        sigma = jnp.exp(self.log_sigma)
        c = self.centers
        phi = phi_lin[..., :, None]
        plus = jnp.exp(-0.5 * ((phi - c) / sigma) ** 2)
        minus = jnp.exp(-0.5 * ((phi + c) / sigma) ** 2)
        return plus - minus


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
