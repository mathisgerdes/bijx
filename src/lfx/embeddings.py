import typing as tp

import jax.numpy as jnp
import numpy as np
from flax import nnx

from .utils import Const


def rescale_range(val, val_range: tuple[float, float] | None):
    if val_range is None:
        return val
    val_min, val_max = val_range
    val = (val - val_min) / (val_max - val_min)
    return val


class Embedding(nnx.Module):
    def __init__(
        self,
        feature_count: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.feature_count = feature_count


class KernelGauss(Embedding):
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
        """
        Smooth interpolation based on Gaussians.

        Given a value x, the output of the kernel function is an array
        roughly like ``[exp(-(x-s1)^2), exp(-(x-s2)^2), ...]``.
        This can be understood as a smooth approximation to linear
        interpolation based on Gaussians located and positions s1, s2, etc.
        These positions are evenly spaced and fixed, here.

        Args:
            feature_count: Number of positions/Gaussians in kernel.
            val_range: Range of input values to rescale to [0, 1].
            width_factor: Initial width factor of the Gaussians.
                The smaller the factor, the wider the Gaussians.
            adaptive_width: Whether to make the width trainable.
            norm: Whether to keep the sum of the kernel values fixed to 1
                for each input value.
            one_width: Whether the widths, if trainable, can be different
                for each kernel position.
            name: Name of module.
        """
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range
        self.width_factor = width_factor
        self.adaptive_width = adaptive_width
        self.norm = norm
        self.one_width = one_width

        width_shape = () if self.one_width else (self.feature_count,)
        if self.adaptive_width:
            self.width_factor = nnx.Param(jnp.full(width_shape, self.width_factor))
        else:
            self.width_factor = Const(width_factor)

    def __call__(self, val):
        factor = nnx.softplus(self.width_factor)
        inverse_width = factor * (self.feature_count - 1)
        # could also make this adaptive
        pos = jnp.linspace(0, 1, self.feature_count)
        val = rescale_range(val, self.val_range)
        val = -((val - pos) ** 2) * inverse_width
        out = jnp.exp(val)
        return out / jnp.sum(out) if self.norm else out


class KernelLin(Embedding):
    def __init__(
        self,
        feature_count: int,
        *,
        val_range: tuple[float, float] | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """Linear interpolation kernel.

        The output of the model is an array like ``[a1, a2, ...]``
        where either one or two neighboring entries are non-zero.
        The position of the non-zero entry is given by the input value
        with linear interpolation (and thus two entries non-zero)
        if the input value falls between two array indices.
        The value the first and last indices correspond to are either
        0 and 1 or given by the ``val_range`` argument.

        Args:
            feature_count: Number of elements in linear interpolation.
            val_range: Range of input values.
            rngs: Random number generators.
        """
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range

    def __call__(self, val):
        width = 1 / (self.feature_count - 1)
        pos = np.linspace(0, 1, self.feature_count)
        val = rescale_range(val, self.val_range)
        val = 1 - jnp.abs(val - pos) / width
        return jnp.maximum(val, 0)


class KernelFourier(Embedding):
    def __init__(
        self,
        feature_count: int,
        *,
        val_range: tuple[float, float] | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """Truncated fourier expansion on given interval.

        Given an input x to the model, the output is an array like
                ``[1, sin(2 pi x), cos(2 pi x), sin(4 pi x), ...]``
            (except in a different order).

        Args:
            feature_count: The number of Fourier-terms.
                (This is true if the number is odd. Otherwise, in effect,
                the next smallest odd number is used).
            val_range: The range of input values such that the largest input
                is normalized to 1.
            rngs: Random number generators.
        """
        super().__init__(feature_count, rngs=rngs)
        self.val_range = val_range

    def __call__(self, val):
        freq = jnp.arange(1, (self.feature_count - 1) // 2 + 1)
        val = rescale_range(val, self.val_range)
        sin = jnp.sin(2 * jnp.pi * freq * val)
        cos = jnp.cos(2 * jnp.pi * freq * val)
        return jnp.concatenate((sin, cos, jnp.array([1.0])))


class KernelReduced(Embedding):
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

        self.superposition = nnx.Param(
            init(rngs.params(), (feature_count, self.kernel.feature_count), jnp.float32)
        )

    def __call__(self, val):
        embed = self.kernel(val)
        sup = self.superposition.value / self.kernel.feature_count
        return jnp.einsum("ij,...j->...i", sup, embed)


class PositionalEmbedding(Embedding):
    """Sinusoidal positional embeddings.

    This embedding is based on the sinusoidal embeddings from Fairseq.
    It maps a scalar value to a vector of sinusoidal embeddings.

    Args:
        feature_count: The number of features in the embedding.
        scale: Scaling factor for the input values (default: 1000).
        rngs: Random number generators.
    """

    def __init__(
        self,
        feature_count: int,
        *,
        scale: float = 1000.0,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(feature_count, rngs=rngs)
        self.scale = scale

    def __call__(self, val):
        t_shape = jnp.shape(val)
        t = jnp.reshape(val, -1)
        t *= self.scale

        half_dim = self.feature_count // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp((-emb) * jnp.arange(half_dim, dtype=t.dtype))
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        if self.feature_count % 2 == 1:  # zero pad
            emb = jnp.pad(emb, ((0, 0), (0, 1)), constant_values=0.0)
        return emb.reshape(*t_shape, self.feature_count)
