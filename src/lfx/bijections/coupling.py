"""
Coupling layer bijections.
"""

from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import auto_vmap

from ..utils import Const
from .base import Bijection


def checker_mask(shape, parity: bool):
    """Checkerboard mask.

    Args:
        shape: Spacial dimensions of input.
        parity: Parity of mask.
    """
    idx_shape = np.ones_like(shape)
    idc = []
    for i, s in enumerate(shape):
        idx_shape[i] = s
        idc.append(np.arange(s, dtype=np.uint8).reshape(idx_shape))
        idx_shape[i] = 1
    mask = (sum(idc) + parity) % 2
    return mask


class AffineCoupling(Bijection):
    """
    Affine coupling layer.

    Masking here is done by multiplication, not by indexing.

    Example:
    ```python
    space_shape = (16, 16)  # no channel/feature dim (add dummy axis below)

    affine_flow = lfx.Chain([
        lfx.ExpandDims(),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, 2, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), True)),
        lfx.AffineCoupling(
            lfx.SimpleConvNet(1, 2, rngs=rngs),
            lfx.checker_mask(space_shape + (1,), False)),
        lfx.ExpandDims().invert(),
    ])
    ```

    `net` should map: `x_f -> act`
    such that `s, t = split(act, 2, -1)`
    and `x_out = t + x_a * exp(s) + x_f`

    Args:
        net: Network that maps frozen features to s, t.
        mask: Mask to apply to input.
    """

    def __init__(self, net: nnx.Module, mask: jax.Array, *, rngs=None):
        self.mask = Const(mask)
        self.net = net

    @property
    def mask_active(self):
        return 1 - self.mask.value

    @property
    def mask_frozen(self):
        return self.mask.value

    def forward(self, x, log_density):
        x_frozen = self.mask_frozen * x
        x_active = self.mask_active * x
        activation = self.net(x_frozen)
        s, t = jnp.split(activation, 2, -1)
        fx = x_frozen + self.mask_active * t + x_active * jnp.exp(s)
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return fx, log_density - log_jac

    def reverse(self, fx, log_density):
        fx_frozen = self.mask_frozen * fx
        fx_active = self.mask_active * fx
        activation = self.net(fx_frozen)
        s, t = jnp.split(activation, 2, -1)
        x = (fx_active - self.mask_active * t) * jnp.exp(-s) + fx_frozen
        axes = tuple(range(-len(self.mask.shape), 0))
        log_jac = jnp.sum(self.mask_active * s, axis=axes)
        return x, log_density + log_jac


class ModuleReconstructor:
    """
    Parameter management utility for dynamically parameterizing modules.

    For convenience, can decompose/reconstruct either modules or states.

    Extracts parameter structure from a module/state and provides methods to
    reconstruct the module from different parameter representations (arrays,
    dicts, leaves). Useful for coupling layers where one network outputs
    parameters for another bijection.

    Representations include:
        - Single array of size `params_total_size`, use `from_array`
        - List of array leaves matching `param_leaves`, use `from_leaves`
        - Dict of params matching `params_dict`, use `from_dict`
        - Full nnx state, use `from_params`
    """

    # params_treedef: Any  # static
    # params_leaves: list[jax.core.ShapedArray]  # static
    # unconditional: nnx.State  # array leaf
    # graph: Any | None = None  # static

    def __init__(
        self, module_or_state: nnx.State | nnx.Module, filter: nnx.Param = nnx.Param
    ):
        if isinstance(module_or_state, nnx.State):
            self.graph = None
            state = module_or_state
        else:
            graph, state = nnx.split(module_or_state)
            self.graph = graph

        params, unconditional = nnx.split_state(state, filter, ...)

        params = jax.tree.map(lambda x: jax.core.ShapedArray(x.shape, x.dtype), params)

        params_leaves, params_treedef = jax.tree.flatten(params)

        self.params_treedef = params_treedef
        self.params_leaves = params_leaves
        self.unconditional = unconditional

    def _tree_flatten(self):
        children = (self.unconditional,)
        aux_data = (self.params_treedef, self.params_leaves, self.graph)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        self = object.__new__(cls)
        self.params_treedef, self.params_leaves, self.graph = aux_data
        (self.unconditional,) = children
        return self

    @property
    def params(self):
        return jax.tree.unflatten(self.params_treedef, self.params_leaves)

    @property
    def params_dict(self):
        return nnx.to_pure_dict(self.params)

    @property
    def params_shapes(self):
        return [p.shape for p in self.params_leaves]

    @property
    def params_shape_dict(self):
        return {k: v.shape for k, v in self.params_dict.items()}

    @property
    def params_dtypes(self):
        return [p.dtype for p in self.params_leaves]

    @property
    def params_sizes(self):
        assert (
            not self.has_complex_params
        ), "Some parameters are complex, need to manually manage these!"
        return [np.prod(s, dtype=int) for s in self.params_shapes]

    @property
    def params_total_size(self):
        return sum(self.params_sizes)

    @property
    def params_array_splits(self):
        return np.cumsum(self.params_sizes)[:-1]

    @property
    def has_complex_params(self):
        return any(np.issubdtype(t, np.complexfloating) for t in self.params_dtypes)

    @property
    def auto_vmap_leaves(self):
        return AutoVmapReconstructor(self, from_leaves=True)

    @property
    def auto_vmap_array(self):
        return AutoVmapReconstructor(self, from_leaves=False)

    @property
    def auto_vmap_dict(self):
        return AutoVmapReconstructor(self, params_shape=self.params_shape_dict)

    def from_params(self, params: nnx.State):
        state = nnx.merge_state(self.unconditional, params)
        if self.graph is None:
            return state
        return nnx.merge(self.graph, state)

    def from_dict(self, params: dict):
        params_state = self.params
        nnx.replace_by_pure_dict(params_state, params)
        return self.from_params(params_state)

    def from_leaves(self, params: list[jax.Array]):
        params = jax.tree.unflatten(self.params_treedef, params)
        return self.from_params(params)

    def from_array(self, params: jax.Array):
        params_leaves = jnp.split(params, self.params_array_splits)
        params_leaves = [
            jnp.reshape(p, s)
            for p, s in zip(params_leaves, self.params_shapes, strict=True)
        ]
        return self.from_leaves(params_leaves)

    def __repr__(self):
        state_or_module = self.params
        if self.graph is not None:
            state_or_module = nnx.merge(self.graph, state_or_module)
        return f"ModuleReconstructor:{state_or_module}"


jax.tree_util.register_pytree_node(
    ModuleReconstructor,
    ModuleReconstructor._tree_flatten,
    ModuleReconstructor._tree_unflatten,
)


@flax.struct.dataclass
class AutoVmapReconstructor:
    """

    Warnings:
        - Need to use param.value inside the module, with implicit value usage
          sometimes get error with vmap nnx.Param is not a valid jax type.
        - Currently keyword arguments cannot be vmap'd over.
    """

    reconstructor: ModuleReconstructor
    from_leaves: bool = False
    params_shape: dict | None = None

    def __call__(
        self, fn_name, params, *args, input_ranks: tuple[int, ...] = (1, 0), **kwargs
    ):

        input_ranks = tuple(input_ranks)
        input_ranks += (None,) * (len(args) - len(input_ranks))

        @auto_vmap(
            self.params_shape if self.params_shape is not None else 1,
            input_ranks,
        )
        def apply(params, args):
            if self.params_shape is not None:
                module = self.reconstructor.from_dict(params)
            elif self.from_leaves:
                module = self.reconstructor.from_leaves(params)
            else:
                module = self.reconstructor.from_array(params)

            fn = getattr(module, fn_name)
            return fn(*args, **kwargs)

        return apply(params, args)

    def __getattr__(self, name: str):
        return partial(self.__call__, name)
