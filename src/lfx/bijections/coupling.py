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


def _indices_to_mask(indices, event_shape):
    mask = jnp.full(event_shape, False)
    mask = mask.at[indices].set(True)
    return mask


class BinaryMask(Bijection):
    """
    Binary mask.

    This class is designed to be a pytree node while providing both
    multiplication-based and indexing-based masking operations.

    Use `from_boolean_mask` or `from_indices` to create a mask.

    Then to apply mask, either use `mask * array` or `array[mask.indices()]`.

    The mask can be flipped with `~mask` or `mask.flip()`.

    For convenience this class can also be used as a bijection,
    which splits the input in the forward pass and merges it in the reverse pass.
    """

    # fundamentally store indices because then indexing is compatible with
    # jax tracers; indexing with boolean mask would yield unknown shapes
    masks: Const
    primary_indices: Const
    secondary_indices: Const
    event_shape: tuple[int, ...]

    def __init__(
        self,
        primary_indices: tuple[np.ndarray, ...],
        event_shape: tuple[int, ...],
        masks: tuple[jax.Array, jax.Array] | None = None,
        secondary_indices: tuple[np.ndarray, ...] | None = None,
    ):
        if masks is None:
            mask = _indices_to_mask(primary_indices, event_shape)
            masks = (mask, ~mask)
        if secondary_indices is None:
            secondary_indices = np.where(masks[1])
        self.masks = Const(masks)
        self.primary_indices = Const(primary_indices)
        self.secondary_indices = Const(secondary_indices)
        self.event_shape = event_shape

    @classmethod
    def from_indices(
        cls, indices: tuple[np.ndarray, ...], event_shape: tuple[int, ...]
    ):
        """Creates a mask from indices."""
        return cls(indices, event_shape)

    @classmethod
    def from_boolean_mask(cls, mask: jax.Array):
        """Creates a mask from a boolean array."""
        return cls(np.where(mask), mask.shape)

    @property
    def boolean_mask(self):
        return self.masks.value[0]

    def indices(
        self, extra_feature_dims: int = 0, batch_safe: bool = True, primary: bool = True
    ):
        ind = (...,) if batch_safe else ()
        ind += self.primary_indices.value if primary else self.secondary_indices.value
        ind += (np.s_[:],) * extra_feature_dims
        return ind

    def flip(self):
        return self.__class__(
            self.secondary_indices.value,
            self.event_shape,
            masks=self.masks.value[::-1],
            secondary_indices=self.primary_indices.value,
        )

    def split(self, array, extra_feature_dims: int = 0, batch_safe: bool = True):
        return (
            array[self.indices(extra_feature_dims, batch_safe, primary=True)],
            array[self.indices(extra_feature_dims, batch_safe, primary=False)],
        )

    def merge(self, primary, secondary, extra_feature_dims: int = 0):
        # Shape analysis: primary is (*batch_dims, num_primary_indices, *feature_dims)
        if extra_feature_dims > 0:
            batch_shape = primary.shape[: -1 - extra_feature_dims]
            feature_shape = primary.shape[-extra_feature_dims:]
        else:
            batch_shape = primary.shape[:-1]
            feature_shape = ()

        # Output shape: (*batch_dims, *event_shape, *feature_dims)
        output_shape = batch_shape + self.event_shape + feature_shape
        output = jnp.zeros(output_shape, dtype=primary.dtype)

        primary_idx = self.indices(extra_feature_dims, batch_safe=True, primary=True)
        secondary_idx = self.indices(extra_feature_dims, batch_safe=True, primary=False)

        output = output.at[primary_idx].set(primary)
        output = output.at[secondary_idx].set(secondary)

        return output

    def forward(self, x, log_density):
        return self.split(x), log_density

    def reverse(self, x, log_density):
        return self.merge(x[0], x[1]), log_density

    # override unary ~ operator
    def __invert__(self):
        return self.flip()

    def __mul__(self, array: jax.Array):
        return self.boolean_mask * array

    def __rmul__(self, array: jax.Array):
        if jnp.ndim(array) < len(self.event_shape):
            # numpy automatically tries to vectorize multiplication;
            # this does not happen with jax arrays
            raise ValueError("rank too low for multiplying by mask (try mask * array)")
        return self.__mul__(array)


def checker_mask(shape, parity: bool):
    """Checkerboard mask.

    Args:
        shape: Spacial dimensions of input.
        parity: Parity of mask.

    Returns:
        BinaryMask instance with checkerboard pattern.
    """
    idx_shape = np.ones_like(shape)
    idc = []
    for i, s in enumerate(shape):
        idx_shape[i] = s
        idc.append(np.arange(s, dtype=np.uint8).reshape(idx_shape))
        idx_shape[i] = 1
    mask = (sum(idc) + parity) % 2
    return BinaryMask.from_boolean_mask(mask.astype(bool))


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
        mask: BinaryMask to apply to input.
    """

    def __init__(self, net: nnx.Module, mask: BinaryMask, *, rngs=None):
        self.mask = mask
        self.net = net

    @property
    def mask_active(self):
        return ~self.mask

    @property
    def mask_frozen(self):
        return self.mask

    def forward(self, x, log_density):
        x_frozen = self.mask_frozen * x
        x_active = self.mask_active * x
        activation = self.net(x_frozen)
        s, t = jnp.split(activation, 2, -1)
        fx = x_frozen + (self.mask_active * t) + x_active * jnp.exp(s)
        axes = tuple(range(-len(self.mask.event_shape), 0))
        log_jac = jnp.sum((self.mask_active * s), axis=axes)
        return fx, log_density - log_jac

    def reverse(self, fx, log_density):
        fx_frozen = self.mask_frozen * fx
        fx_active = self.mask_active * fx
        activation = self.net(fx_frozen)
        s, t = jnp.split(activation, 2, -1)
        x = (fx_active - (self.mask_active * t)) * jnp.exp(-s) + fx_frozen
        axes = tuple(range(-len(self.mask.event_shape), 0))
        log_jac = jnp.sum((self.mask_active * s), axis=axes)
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
        return jax.tree.map(jnp.shape, self.params_dict)

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

    def from_state(self, params: nnx.State):
        state = nnx.merge_state(self.unconditional, params)
        if self.graph is None:
            return state
        return nnx.merge(self.graph, state)

    def _params_rank(self, params: dict | list[jax.Array] | jax.Array | nnx.State):
        if isinstance(params, nnx.State):
            return jax.tree.map(jnp.ndim, self.params)
        if isinstance(params, dict):
            return jax.tree.map(jnp.ndim, self.params_dict)
        if isinstance(params, list):
            return [jnp.ndim(p) for p in self.params_leaves]
        if isinstance(params, jax.Array):
            return 1  # always flattened
        raise TypeError(f"Unsupported parameter type: {type(params)}")

    def from_parameters(
        self,
        params: dict | list[jax.Array] | jax.Array | nnx.State,
        auto_vmap: bool = False,
    ):
        """Reconstructs the module from different parameter representations.

        This method dispatches to the correct reconstruction logic based on the
        input type.

        Args:
            params: Can be a single array, a list of arrays, a dict, or a
                full nnx state.
        """
        if auto_vmap:
            return AutoVmapReconstructor(
                self,
                params,
                params_rank=self._params_rank(params),
            )

        if isinstance(params, nnx.State):
            return self.from_state(params)

        if isinstance(params, dict):
            params_state = self.params
            nnx.replace_by_pure_dict(params_state, params)
            return self.from_state(params_state)

        if isinstance(params, list):
            unflattened_params = jax.tree.unflatten(self.params_treedef, params)
            return self.from_state(unflattened_params)

        if isinstance(params, jax.Array):
            params_leaves = jnp.split(params, self.params_array_splits)
            params_leaves = [
                jnp.reshape(p, s)
                for p, s in zip(params_leaves, self.params_shapes, strict=True)
            ]
            unflattened_params = jax.tree.unflatten(self.params_treedef, params_leaves)
            return self.from_state(unflattened_params)

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
    """Wrap reconstruction + function call with vmap to support batching.

    Warnings:
        - Need to use param.value inside the module, with implicit value usage
          sometimes get error with vmap nnx.Param is not a valid jax type.
        - Currently keyword arguments cannot be vmap'd over.
    """

    reconstructor: ModuleReconstructor
    params: nnx.State | dict | list[jax.Array] | jax.Array
    params_rank: int | dict = 1

    def __call__(self, fn_name, *args, input_ranks: tuple[int, ...] = (1, 0), **kwargs):

        input_ranks = tuple(input_ranks)
        input_ranks += (None,) * (len(args) - len(input_ranks))

        @auto_vmap(
            self.params_rank,
            input_ranks,
        )
        def apply(params, args):
            module = self.reconstructor.from_parameters(params)
            fn = getattr(module, fn_name)
            return fn(*args, **kwargs)

        return apply(self.params, args)

    def __getattr__(self, name: str):
        return partial(self.__call__, name)


class GeneralCouplingLayer(Bijection):

    def __init__(
        self,
        embedding_net: nnx.Module,
        mask: BinaryMask,
        bijection_reconstructor: ModuleReconstructor,
        split: bool = True,  # if false, use masking by multiplication
    ):
        self.embedding_net = embedding_net
        self.mask = mask
        self.bijection_reconstructor = bijection_reconstructor
        self.split = split

    def _split(self, x):
        if self.split:
            return self.mask.split(x)
        else:
            return self.mask * x, ~self.mask * x

    def _merge(self, active, passive):
        if self.split:
            return self.mask.merge(active, passive)
        else:
            # assume passive was not modified; no need to mask again
            # active masked for safety (in case passive part was modified)
            return self.mask * active + passive

    def _apply(self, x, log_density, inverse=False, **kwargs):
        active, passive = self._split(x)

        params = self.embedding_net(passive)
        bijection = self.bijection_reconstructor.from_parameters(params, auto_vmap=True)

        method = bijection.reverse if inverse else bijection.forward
        active, delta_log_density = method(
            active, jnp.zeros_like(log_density), input_ranks=(1, 0)
        )

        if not self.split:
            delta_log_density *= self.mask
            axes = tuple(range(-len(self.mask.event_shape), 0))
            delta_log_density = jnp.sum(delta_log_density, axis=axes)
        else:
            event_axes = tuple(
                range(jnp.ndim(log_density), jnp.ndim(delta_log_density))
            )
            delta_log_density = jnp.sum(delta_log_density, axis=event_axes)

        log_density += delta_log_density

        x = self._merge(active, passive)

        return x, log_density

    def forward(self, x, log_density, **kwargs):
        return self._apply(
            x,
            log_density,
            inverse=False,
            **kwargs,
        )

    def reverse(self, x, log_density, **kwargs):
        return self._apply(
            x,
            log_density,
            inverse=True,
            **kwargs,
        )
