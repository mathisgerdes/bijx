import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class NonlinearFeatures(nnx.Module):

    def __init__(
        self,
        out_channel_size: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Base class for non-linear feature mappings."""
        self.out_channel_size = out_channel_size

    def apply_feature_map(self, inputs, **kwargs):
        raise NotImplementedError()

    def __call__(
        self, inputs, local_coupling, flatten_features=True, mask=None, **kwargs
    ):
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
        features = jnp.einsum("...i,ij->...ij", phi_lin, self.phi_freq.value)
        features = jnp.sin(features)
        return features


class PolynomialFeatures(NonlinearFeatures):
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
        features = jnp.stack([phi_lin**p for p in self.powers], axis=-1)
        return features


class DivFeatures(NonlinearFeatures):
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
            freq_init(
                rngs.params(),
                (input_channels, feature_count),
            )
        )

    def apply_feature_map(self, phi_lin, **kwargs):
        freq = jnp.abs(self.phi_freq)
        features = jnp.einsum("...i,ij->...ij", -(phi_lin**2), 1 / freq)
        features = jnp.einsum("...ij,...i,ij->...ij", jnp.exp(features), phi_lin, freq)
        return features


class ConcatFeatures(NonlinearFeatures):
    def __init__(
        self,
        features: list[NonlinearFeatures],
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(sum(f.out_channel_size for f in features), rngs=rngs)
        self.features = features

    def apply_feature_map(self, phi_lin, **kwargs):
        return jnp.concatenate(
            [f.apply_feature_map(phi_lin, **kwargs) for f in self.features], axis=-1
        )
