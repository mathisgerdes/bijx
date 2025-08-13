r"""
Embedding layers for time and positional encoding in neural networks.

This module provides various embedding functions that map scalar inputs
to high-dimensional feature vectors. These are particularly useful for
continuous normalizing flows where time-dependent parameters need rich
feature representations, and for positional encodings in attention mechanisms.
"""

import typing as tp

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import autovmap

from ..utils import Const


def rescale_range(val, val_range: tuple[float, float] | None):
    """Rescale input values to unit interval [0, 1].

    Args:
        val: Input values to rescale.
        val_range: Tuple of (min, max) values defining the input range.
            If None, returns input unchanged.

    Returns:
        Rescaled values mapped to [0, 1] interval.

    Note:
        Values outside the specified range will be mapped outside [0, 1].
        Consider clamping if strict bounds are required.
    """
    if val_range is None:
        return val
    val_min, val_max = val_range
    val = (val - val_min) / (val_max - val_min)
    return val


class Embedding(nnx.Module):
    """Base class for scalar-to-vector embedding functions.

    Provides the foundation for all embedding layers that map scalar inputs
    to fixed-size feature vectors. Subclasses implement specific embedding
    strategies like Gaussian kernels, Fourier features, or positional encodings.

    Args:
        feature_count: Dimensionality of the output feature vector.
        rngs: Random number generator state for parameter initialization.

    Note:
        This is an abstract base class. Use concrete subclasses like
        :class:`KernelGauss`, :class:`KernelFourier`, or :class:`PositionalEmbedding`.
        Its main function is to ensure the ``feature_count`` parameter is set.
    """

    def __init__(
        self,
        feature_count: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.feature_count = feature_count


class KernelGauss(Embedding):
    r"""Gaussian radial basis function embedding with learnable widths.

    Maps scalar inputs to feature vectors using Gaussian basis functions
    centered at evenly spaced positions.

    The centers $\mu_i$ are evenly spaced across the input range, and the
    width parameter $\sigma$ can be learned during training for optimal
    feature representation. The width parameter is passed through softplus
    to ensure positivity.

    Args:
        feature_count: Number of Gaussian basis functions.
        val_range: Input value range for rescaling to [0, 1].
        width_factor: Initial width parameter (smaller = wider Gaussians).
        adaptive_width: Whether to make width parameters trainable.
        norm: Whether to normalize outputs to sum to 1 (probability-like).
        one_width: If adaptive, whether to use single width vs per-Gaussian widths.
        rngs: Random number generator state.

    Example:
        >>> # Smooth time embedding for CNFs
        >>> embed = KernelGauss(feature_count=21, adaptive_width=True, rngs=rngs)
        >>> features = embed(0.1)  # Shape: (21,)
    """

    def __init__(
        self,
        feature_count: int,
        *,
        val_range: tuple[float, float] | None = None,
        width_factor: float = np.log(np.exp(1) - 1),
        adaptive_width: bool = True,
        norm: bool = True,
        one_width: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range
        self.adaptive_width = adaptive_width
        self.norm = norm
        self.one_width = one_width

        width_shape = () if self.one_width else (self.feature_count,)
        if self.adaptive_width:
            self.width_factor = nnx.Param(jnp.full(width_shape, width_factor))
        else:
            self.width_factor = Const(width_factor)

    @autovmap(val=0)
    def __call__(self, val):
        """Apply Gaussian basis function embedding to input values.

        Args:
            val: Input scalar values of any shape.

        Returns:
            Gaussian feature activations with shape (*val.shape, feature_count).
        """
        factor = nnx.softplus(self.width_factor)
        inverse_width = factor * (self.feature_count - 1)
        # could also make this adaptive
        pos = jnp.linspace(0, 1, self.feature_count)
        val = rescale_range(val, self.val_range)
        val = -((val - pos) ** 2) * inverse_width
        out = jnp.exp(val)
        return out / jnp.sum(out) if self.norm else out


class KernelLin(Embedding):
    """Piecewise linear interpolation embedding with sparse outputs.

    Maps scalar inputs to sparse feature vectors using linear interpolation
    between adjacent basis positions. At most two adjacent features are
    non-zero for any input.

    Args:
        feature_count: Number of interpolation basis positions.
        val_range: Input value range for rescaling to [0, 1].
        rngs: Random number generator state.

    Example:
        >>> # Efficient sparse embedding
        >>> embed = KernelLin(feature_count=11, rngs=rngs)
        >>> features = embed(t)  # Only 2 non-zero values
    """

    def __init__(
        self,
        feature_count: int,
        *,
        val_range: tuple[float, float] | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range

    @autovmap(val=0)
    def __call__(self, val):
        """Apply linear interpolation embedding to input values.

        Args:
            val: Input scalar values of any shape.

        Returns:
            Sparse linear interpolation features with shape (*val.shape, feature_count).
        """
        width = 1 / (self.feature_count - 1)
        pos = np.linspace(0, 1, self.feature_count)
        val = rescale_range(val, self.val_range)
        val = 1 - jnp.abs(val - pos) / width
        return jnp.maximum(val, 0)


class KernelFourier(Embedding):
    r"""Fourier series embedding for smooth representations.

    Maps scalar inputs to feature vectors using truncated Fourier series
    with sine and cosine components.
    The embedding captures multiple frequency scales, allowing the network
    to represent both fine-grained and coarse temporal patterns effectively.

    Args:
        feature_count: Number of Fourier terms in the expansion.
            For even counts, uses (feature_count-1)//2 frequency pairs plus constant.
        val_range: Input value range for normalization.
        rngs: Random number generator state.

    Note:
        The constant term (1.0) is always included.
        Frequencies increase linearly: 1, 2, 3, ... in units of 2π/period.

    Example:
        >>> embed = KernelFourier(feature_count=21, rngs=rngs)
        >>> time_features = embed(t)  # Captures multiple time scales
    """

    def __init__(
        self,
        feature_count: int,
        *,
        val_range: tuple[float, float] | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range

    @autovmap(val=0)
    def __call__(self, val):
        """Apply Fourier series embedding to input values.

        Args:
            val: Input scalar values of any shape.

        Returns:
            Fourier feature vector with shape (*val.shape, feature_count).
        """
        freq = jnp.arange(1, (self.feature_count - 1) // 2 + 1)
        val = rescale_range(val, self.val_range)
        sin = jnp.sin(2 * jnp.pi * freq * val)
        cos = jnp.cos(2 * jnp.pi * freq * val)
        return jnp.concatenate((sin, cos, jnp.array([1.0])))


class KernelReduced(Embedding):
    """Dimensionality reduction wrapper for high-dimensional embeddings.

    Applies learned linear dimensionality reduction to another embedding layer.
    Note that it simply implements a linear map, not strictly a "reduction"
    and the output features could be chosen larger than the input features.

    Args:
        kernel: Base embedding layer to reduce.
        feature_count: Target dimensionality (must be < kernel.feature_count).
        init: Initialization function for projection matrix (default: orthogonal).
        rngs: Random number generator state.

    Note:
        The projection matrix is normalized by the base kernel's feature count
        to maintain reasonable magnitudes. Orthogonal initialization helps
        preserve independent features.

    Example:
        >>> # Reduce 49-dimensional Fourier embedding to 20 dimensions
        >>> base_embed = KernelFourier(49, rngs=rngs)
        >>> reduced = KernelReduced(base_embed, 20, rngs=rngs)
        >>> features = reduced(t)
    """

    def __init__(
        self,
        kernel: nnx.Module,
        feature_count: int,
        *,
        init: tp.Callable = nnx.initializers.orthogonal(),
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        self.kernel = kernel

        chosen_param_dtype = jnp.result_type(0.0)
        self.superposition = nnx.Param(
            init(
                rngs.params(),
                (feature_count, self.kernel.feature_count),
                chosen_param_dtype,
            )
        )

    def __call__(self, val):
        """Apply dimensionality reduction to base kernel embedding.

        Args:
            val: Input scalar values.

        Returns:
            Reduced-dimension embedding features.
        """
        embed = self.kernel(val)
        sup = self.superposition.value / self.kernel.feature_count
        return jnp.einsum("ij,...j->...i", sup, embed)


class PositionalEmbedding(Embedding):
    r"""Sinusoidal positional embeddings from transformer architectures.

    Uses multiple frequencies to create position-dependent representations.

    Args:
        feature_count: Output feature dimensionality (must be even).
        max_positions: Maximum expected input value for frequency scaling.
            Controls the base wavelength; larger values create lower frequencies.
        append_input: Whether to concatenate the raw input to the embedding.
        rngs: Random number generator state.

    Example:
        >>> embed = PositionalEmbedding(feature_count=64, max_positions=1000, rngs=rngs)
        >>> pos_features = embed(position_indices)
    """

    def __init__(
        self,
        feature_count: int,
        *,
        max_positions: float = 10000.0,
        append_input: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        assert feature_count % 2 == 0, "feature_count must be even"
        self.max_positions = max_positions
        self.append_input = append_input

    @autovmap(val=0)
    def __call__(self, val):
        """Apply sinusoidal positional embedding to input values.

        Args:
            val: Scalar input values of any shape.

        Returns:
            Positional embeddings with shape (*val.shape, feature_count + append_input).
        """
        val_shape = jnp.shape(val)
        val = jnp.reshape(val, -1)

        half_dim = self.feature_count // 2
        emb = np.log(self.max_positions) / (half_dim - 1)
        emb = jnp.exp((-emb) * jnp.arange(half_dim, dtype=val.dtype))
        emb = val[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        # if self.feature_count % 2 == 1:  # zero pad
        #     emb = jnp.pad(emb, [[0, 0], [0, 1]])
        if self.append_input:
            emb = jnp.concatenate([emb, val[:, None]], axis=1)
        return emb.reshape(*val_shape, self.feature_count + self.append_input)
