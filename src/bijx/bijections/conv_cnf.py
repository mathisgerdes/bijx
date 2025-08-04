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
    if par.value is None:
        return None
    return rearrange(par.value, "... (c t) -> ... c t", t=t_emb.shape[-1]) @ t_emb


class ConvCNF(nnx.Module):
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

        conv_params["kernel_params"].value = _contract_with_emb(
            conv_params["kernel_params"], t_emb
        )
        conv_params["bias"].value = _contract_with_emb(conv_params["bias"], t_emb)

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
