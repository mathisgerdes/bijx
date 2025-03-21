import typing as tp
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from . import embeddings
from .bijections import Const
from .conv import ConvSym, kernel_d4
from .utils import ShapeInfo


class NonlinearFeatures(nnx.Module):

    def __init__(
            self,
            out_channel_size: int,
            *,
            rngs: nnx.Rngs | None = None,
        ):
        """Base class for non-linear feature mappings."""
        self.out_channel_size = out_channel_size

    def apply_feature_map(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs, local_coupling, flatten_features=True):
        # Compute divergence using local couplings (W_xx part of conv kernel)
        # Feature map is exclusively site-wise
        orig_channels = inputs.shape[-1]
        inputs, bwd = jax.vjp(self.apply_feature_map, inputs)

        if flatten_features:
            local_coupling = local_coupling.reshape(
                orig_channels, orig_channels, -1)

        idc = np.arange(local_coupling.shape[1])
        cotangent_reshape = (*inputs.shape[:-2], 1, 1)
        cotangent = jnp.tile(local_coupling[idc, idc], cotangent_reshape)

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

        # Initialize parameters in __init__
        self.phi_freq = nnx.Param(
            freq_init(
                rngs.params(),
                (input_channels, feature_count)
            )
        )

    def apply_feature_map(self, phi_lin):
        features = jnp.einsum('...i,ij->...ij', phi_lin, self.phi_freq.value)
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

    def apply_feature_map(self, phi_lin):
        features = jnp.stack([phi_lin ** p for p in self.powers], axis=-1)
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

        # Initialize parameters in __init__
        self.phi_freq = nnx.Param(
            freq_init(
                rngs.params(),
                (input_channels, feature_count),
            )
        )

    def apply_feature_map(self, phi_lin):
        freq = jnp.abs(self.phi_freq)
        features = jnp.einsum('...i,ij->...ij', -phi_lin**2, 1/freq)
        features = jnp.einsum('...ij,...i,ij->...ij', jnp.exp(features), phi_lin, freq)
        return features


class ConcatFeatures(NonlinearFeatures):
    def __init__(
            self,
            features: list[NonlinearFeatures],
            rngs: nnx.Rngs | None = None,
        ):
        super().__init__(sum(f.out_channel_size for f in features), rngs=rngs)
        self.features = features

    def apply_feature_map(self, phi_lin):
        return jnp.concatenate([f.apply_feature_map(phi_lin) for f in self.features], axis=-1)


def _contract_with_emb(par, t_emb):
    if par.value is None:
        return None
    return rearrange(
        par.value,
        '... (c t) -> ... c t',
        t=t_emb.shape[-1]
    ) @ t_emb


class Phi4CNF(nnx.Module):
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
        batch_shape, shape_info = self.shape_info.process_event(x.shape)
        channel_size = shape_info.channel_size
        x = x.reshape(-1, *shape_info.space_shape, channel_size)

        t_emb = self.time_kernel(t)

        # contract time embedding with conv kernel & bias
        conv_graph, conv_params = nnx.split(self.conv)

        conv_params['kernel_params'].value = _contract_with_emb(
            conv_params['kernel_params'], t_emb)
        conv_params['bias'].value = _contract_with_emb(
            conv_params['bias'], t_emb)

        conv = nnx.merge(conv_graph, conv_params)

        feature_superposition = (
            self.feature_superposition.value
            / self.feature_map.out_channel_size
        )

        # extract the local-coupling weights; shape=(in features, out features)
        w00 = conv.kernel_params.value[0]
        # contract with feature superposition
        w00 = jnp.einsum('if,io->fo', feature_superposition, w00)

        features, div = self.feature_map(x, w00)
        features = jnp.einsum('fw,...w->...f', feature_superposition, features)
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
            time_kernel_reduced = 20,
            features: tuple[NonlinearFeatures, ...] = (
                partial(FourierFeatures, 49),
                partial(PolynomialFeatures, (1,)),
            ),
            features_reduced: int | None = 20,
            rngs: nnx.Rngs,
        ):
        channel_size = np.prod(channel_shape, dtype=int)

        if time_kernel_reduced is not None:
            time_kernel = embeddings.KernelReduced(time_kernel, time_kernel_reduced, rngs=rngs)

        features = ConcatFeatures([
            f_map(channel_size, rngs=rngs)
            for f_map in features
        ])

        conv_in_features = features.out_channel_size
        if features_reduced is not None:
            feature_superposition = nnx.Param(
                nnx.initializers.orthogonal()(
                    rngs.params(),
                    (features_reduced, features.out_channel_size)
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

        shape_info = ShapeInfo(space_dim=len(kernel_shape), channel_dim=len(channel_shape))

        return cls(
            shape_info=shape_info,
            conv=conv,
            time_kernel=time_kernel,
            feature_map=features,
            feature_superposition=feature_superposition,
        )
