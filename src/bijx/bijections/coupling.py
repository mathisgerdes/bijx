"""
Coupling layer bijections.
"""

import functools
import inspect

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

    @property
    def count_primary(self):
        return self.primary_indices.value[0].size

    @property
    def count_secondary(self):
        return self.secondary_indices.value[0].size

    @property
    def counts(self):
        return self.count_primary, self.count_secondary

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
        if isinstance(params, np.ndarray | jax.Array):
            return 1  # always flattened
        raise TypeError(f"Unsupported parameter type: {type(params)}")

    def from_params(
        self,
        params: dict | list[jax.Array] | jax.Array | nnx.State,
        auto_vmap: bool = False,
    ):
        """Reconstructs the module from different parameter representations.

        This method dispatches to the correct reconstruction logic based on the
        input type.

        If auto_vmap is True, an object is returned that behaves almost like
        the module except that function calls are automatically vectorized
        (via vmap) over parameters and inputs.

        Args:
            params: Can be a single array, a list of arrays, a dict, or a
                full nnx state.
            auto_vmap: If True, wrap the reconstruction in an AutoVmapReconstructor.
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

        if isinstance(params, np.ndarray | jax.Array):
            params_leaves = jnp.split(params, self.params_array_splits, -1)
            params_leaves = [
                jnp.reshape(p, p.shape[:-1] + s)
                for p, s in zip(params_leaves, self.params_shapes, strict=True)
            ]
            unflattened_params = jax.tree.unflatten(self.params_treedef, params_leaves)
            return self.from_state(unflattened_params)

        raise TypeError(f"Unsupported parameter type: {type(params)}")

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

    def __call__(self, fn_name, *args, input_ranks: tuple[int, ...] = (0, 0), **kwargs):

        input_ranks = tuple(input_ranks)
        input_ranks += (None,) * (len(args) - len(input_ranks))

        @auto_vmap(
            self.params_rank,
            input_ranks,
        )
        def apply(params, args):
            module = self.reconstructor.from_params(params)
            fn = getattr(module, fn_name)
            return fn(*args, **kwargs)

        return apply(self.params, args)

    def __getattr__(self, name: str):
        module = self.reconstructor.from_params(self.params)
        bare_attr = getattr(module, name)

        if not callable(bare_attr):
            return bare_attr

        # Create a wrapper function
        original_sig = inspect.signature(bare_attr)
        new_params = list(original_sig.parameters.values())

        # Insert input_ranks before any VAR_KEYWORD parameter
        input_ranks_param = inspect.Parameter(
            "input_ranks",
            inspect.Parameter.KEYWORD_ONLY,
            default=(0, 0),
            annotation=tuple[int, ...],
        )

        # Find position to insert (before VAR_KEYWORD if it exists)
        insert_pos = len(new_params)
        for i, param in enumerate(new_params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                insert_pos = i
                break

        new_params.insert(insert_pos, input_ranks_param)
        new_sig = original_sig.replace(parameters=new_params)

        def wrapped(*args, input_ranks=(0, 0), **kwargs):
            return self.__call__(name, *args, input_ranks=input_ranks, **kwargs)

        wrapped = functools.update_wrapper(wrapped, bare_attr)
        wrapped.__signature__ = new_sig
        return wrapped

    def __repr__(self):
        state_or_module = self.reconstructor.from_params(self.params)
        return f"AutoVmapReconstructor:{state_or_module}"


class GeneralCouplingLayer(Bijection):

    def __init__(
        self,
        embedding_net: nnx.Module,
        mask: BinaryMask,
        bijection_reconstructor: ModuleReconstructor,
        bijection_event_rank: int = 0,
        split: bool = True,  # if false, use masking by multiplication
    ):
        self.embedding_net = embedding_net
        self.mask = mask
        self.bijection_reconstructor = bijection_reconstructor
        self.split = split
        self.bijection_event_rank = bijection_event_rank

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

        active_rank = self.bijection_event_rank
        if self.split:
            if active_rank == 0:
                dens_shape = jnp.shape(log_density) + (1,)
            elif active_rank == 1:
                dens_shape = jnp.shape(log_density)
            else:
                raise ValueError(
                    "Split reduces active and passive arrays to be vectors; "
                    "bijection_event_rank must be 0 or 1"
                )
        else:
            if len(self.mask.event_shape) < active_rank:
                raise ValueError(
                    f"Event rank given mask shape {self.mask.event_shape} "
                    f"is too low for bijection_event_rank {self.bijection_event_rank}"
                )
            broadcast_rank = len(self.mask.event_shape) - active_rank
            dens_shape = jnp.shape(log_density) + (1,) * broadcast_rank

        params = self.embedding_net(passive)
        bijection = self.bijection_reconstructor.from_params(params, auto_vmap=True)

        method = bijection.reverse if inverse else bijection.forward
        active, delta_log_density = method(
            active, jnp.zeros(dens_shape), input_ranks=(active_rank, 0)
        )

        if not self.split:
            # sum over event axes that were vmap'd over
            delta_log_density *= self.mask
            axes = tuple(range(-broadcast_rank, 0))
            delta_log_density = jnp.sum(delta_log_density, axis=axes)
        elif active_rank == 0:
            # case: applied vmap over flattened event axes
            delta_log_density = jnp.sum(delta_log_density, axis=-1)

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
