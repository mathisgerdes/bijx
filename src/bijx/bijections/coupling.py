r"""Coupling layer bijections for normalizing flows.

This module implements coupling layers, which are fundamental building blocks
of normalizing flows that maintain tractable Jacobian determinants by
transforming only a subset of input dimensions at a time.

Coupling layers work by:
1. Splitting input into two parts using a binary mask
2. Using one part to parameterize a bijection applied to the other part
3. Keeping the first part unchanged to ensure invertibility

Key components:
- :class:`BinaryMask`: Flexible binary masking with indexing and boolean operations
- :class:`ModuleReconstructor`: Parameter management for dynamic module reconstruction
- :class:`GeneralCouplingLayer`: Full coupling layer implementation with mask management

Note:
    The general coupling layer implemented here permits both bijections that
    fundamentally act on scalar values, as well as more general bijections
    that could transform blocks of values in a way that does not factorize.
"""

import functools
import inspect
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import autovmap

from .base import Bijection


def _indices_to_mask(indices, event_shape):
    """Convert index tuple to boolean mask array."""
    mask = jnp.full(event_shape, False)
    mask = mask.at[indices].set(True)
    return mask


class BinaryMask(Bijection):
    r"""Binary mask for coupling layer split/merge operations.

    This class provides a flexible masking utility that supports both
    multiplication-based and indexing-based masking operations, registered
    as a jax pytree node.

    The mask can be used in two main ways:
    1. As a masking operator: Use ``mask * array`` or ``array[mask.indices()]``
    2. As a bijection: Splits input in forward pass, merges in reverse pass

    Type: $\mathbb{R}^n \to \mathbb{R}^{n_1} \times \mathbb{R}^{n_2}$ (forward)
    Transform: Splits input according to binary mask pattern

    Args:
        primary_indices: Tuple of arrays specifying primary (True) indices.
        event_shape: Shape of the event dimensions being masked.
        masks: Optional precomputed boolean mask pair (primary, secondary).
        secondary_indices: Optional secondary (False) indices.

    The mask can be flipped with ``~mask`` or :meth:`flip()` to swap primary/secondary.
    Use :meth:`from_boolean_mask` or :meth:`from_indices` for convenient construction.

    Example:
        >>> # Create checkerboard mask
        >>> mask = checker_mask((4, 4), parity=0)
        >>> x = jnp.ones((4, 4))
        >>> primary, secondary = mask.split(x)
        >>> reconstructed = mask.merge(primary, secondary)
        >>> # reconstructed == x
    """

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
        self.primary_indices = nnx.data(primary_indices)
        self.event_shape = nnx.static(event_shape)
        self.masks = nnx.data(masks)
        self.secondary_indices = nnx.data(secondary_indices)

    @property
    def count_primary(self):
        """Number of elements in the primary (True) mask region."""
        return self.primary_indices[0].size

    @property
    def count_secondary(self):
        """Number of elements in the secondary (False) mask region."""
        return self.secondary_indices[0].size

    @property
    def counts(self):
        """Tuple of (primary_count, secondary_count)."""
        return self.count_primary, self.count_secondary

    @property
    def event_size(self):
        """Total number of elements in the event shape."""
        return sum(self.counts)

    @classmethod
    def from_indices(
        cls, indices: tuple[np.ndarray, ...], event_shape: tuple[int, ...]
    ):
        """Create mask from index arrays.

        Args:
            indices: Tuple of index arrays specifying primary mask positions.
            event_shape: Shape of the event dimensions being masked.

        Returns:
            New BinaryMask instance.
        """
        return cls(indices, event_shape)

    @classmethod
    def from_boolean_mask(cls, mask: jax.Array):
        """Create mask from boolean array.

        Args:
            mask: Boolean array where True indicates primary mask positions.

        Returns:
            New BinaryMask instance with same pattern as input mask.
        """
        return cls(np.where(mask), mask.shape)

    @property
    def boolean_mask(self):
        """Primary boolean mask array."""
        return self.masks[0]

    def indices(
        self, extra_channel_dims: int = 0, batch_safe: bool = True, primary: bool = True
    ):
        """Get indexing tuple for array access.

        Args:
            extra_channel_dims: Number of trailing channel dimensions to preserve.
            batch_safe: If True, include ellipsis for batch dimensions.
            primary: If True, return primary indices; otherwise secondary.

        Returns:
            Indexing tuple suitable for array subscripting.
        """
        ind = (...,) if batch_safe else ()
        ind += self.primary_indices if primary else self.secondary_indices
        ind += (np.s_[:],) * extra_channel_dims
        return ind

    def flip(self):
        """Create flipped mask with primary/secondary swapped.

        Returns:
            New BinaryMask with primary and secondary regions swapped.
        """
        return self.__class__(
            self.secondary_indices,
            self.event_shape,
            masks=self.masks[::-1],
            secondary_indices=self.primary_indices,
        )

    def split(self, array, extra_channel_dims: int = 0, batch_safe: bool = True):
        """Split array into primary and secondary parts according to mask.

        Args:
            array: Input array to split.
            extra_channel_dims: Number of trailing channel dimensions to preserve.
            batch_safe: If True, handle arbitrary batch dimensions.

        Returns:
            Tuple of (primary_part, secondary_part) arrays.
        """
        return (
            array[self.indices(extra_channel_dims, batch_safe, primary=True)],
            array[self.indices(extra_channel_dims, batch_safe, primary=False)],
        )

    def merge(self, primary, secondary, extra_channel_dims: int = 0):
        """Merge primary and secondary parts back into full array.

        Reconstructs the original array structure by placing primary and
        secondary parts at their respective mask positions.

        Args:
            primary: Primary part array with masked elements.
            secondary: Secondary part array with complementary elements.
            extra_channel_dims: Number of trailing channel dimensions.

        Returns:
            Merged array with original event shape restored.
        """
        # Shape analysis: primary is (*batch_dims, num_primary_indices, *channel_dims)
        if extra_channel_dims > 0:
            batch_shape = primary.shape[: -1 - extra_channel_dims]
            channel_shape = primary.shape[-extra_channel_dims:]
        else:
            batch_shape = primary.shape[:-1]
            channel_shape = ()

        # Output shape: (*batch_dims, *event_shape, *channel_dims)
        output_shape = batch_shape + self.event_shape + channel_shape
        output = jnp.zeros(output_shape, dtype=primary.dtype)

        primary_idx = self.indices(extra_channel_dims, batch_safe=True, primary=True)
        secondary_idx = self.indices(extra_channel_dims, batch_safe=True, primary=False)

        output = output.at[primary_idx].set(primary)
        output = output.at[secondary_idx].set(secondary)

        return output

    def forward(self, x, log_density):
        """Split input as bijection forward pass.

        When used as a bijection, forward pass splits the input into
        primary and secondary parts according to the mask.

        Args:
            x: Input array to split.
            log_density: Input log density (unchanged).

        Returns:
            Tuple of ((primary_part, secondary_part), unchanged_log_density).
        """
        return self.split(x), log_density

    def reverse(self, x, log_density):
        """Merge parts as bijection reverse pass.

        When used as a bijection, reverse pass merges the split parts
        back into the original array structure.

        Args:
            x: Tuple of (primary_part, secondary_part) to merge.
            log_density: Input log density (unchanged).

        Returns:
            Tuple of (merged_array, unchanged_log_density).
        """
        return self.merge(x[0], x[1]), log_density

    # override unary ~ operator
    def __invert__(self):
        """Flip mask using ~ operator (equivalent to flip())."""
        # note that this is distinct from mask.invert(), as a bijection
        return self.flip()

    def __mul__(self, array: jax.Array):
        """Element-wise multiplication with boolean mask.

        Args:
            array: Array to mask.

        Returns:
            Array with masked elements zeroed out.
        """
        return self.boolean_mask * array

    def __rmul__(self, array: jax.Array):
        """Right multiplication with boolean mask.

        Handles array * mask syntax with appropriate error checking.

        Args:
            array: Array to mask.

        Returns:
            Array with masked elements zeroed out.
        """
        if jnp.ndim(array) < len(self.event_shape):
            # numpy automatically tries to vectorize multiplication;
            # this does not happen with jax arrays
            raise ValueError("rank too low for mask multiplication (try mask * array)")
        return self.__mul__(array)


def checker_mask(shape, parity: bool):
    """Create checkerboard pattern binary mask.

    Generates a checkerboard (alternating) pattern mask commonly used in
    coupling layers for spatial data like images. The pattern alternates
    between True/False based on the sum of coordinates.

    Args:
        shape: Spatial dimensions of the input array.
        parity: Starting parity - if True, (0,0,...) position starts as True.

    Returns:
        :class:`BinaryMask` instance with checkerboard pattern.

    Example:
        >>> # 2x2 checkerboard with parity=0
        >>> mask = checker_mask((2, 2), parity=False)
        >>> mask.boolean_mask.shape
        (2, 2)
    """
    idx_shape = np.ones_like(shape)
    idc = []
    for i, s in enumerate(shape):
        idx_shape[i] = s
        idc.append(np.arange(s, dtype=np.uint8).reshape(idx_shape))
        idx_shape[i] = 1
    mask = (sum(idc) + parity) % 2
    return BinaryMask.from_boolean_mask(mask.astype(bool))


class ModuleReconstructor(nnx.Pytree):
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

    def __init__(
        self,
        module_or_state: nnx.State | nnx.Module,
        filter: nnx.Param = nnx.Param,
    ):
        if isinstance(module_or_state, nnx.State):
            self.graph = None
            state = module_or_state
        else:
            graph, state = nnx.split(module_or_state)
            self.graph = nnx.static(graph)

        params, unconditional = nnx.split_state(state, filter, ...)

        params = jax.tree.map(lambda x: jax.core.ShapedArray(x.shape, x.dtype), params)

        params_leaves, params_treedef = jax.tree.flatten(params)

        self.params_treedef = nnx.static(params_treedef)
        self.params_leaves = nnx.static(params_leaves)
        self.unconditional = nnx.data(unconditional)

    # def _tree_flatten(self):
    #     children = (self.unconditional,)
    #     aux_data = (self.params_treedef, self.params_leaves, self.graph)
    #     return children, aux_data

    # @classmethod
    # def _tree_unflatten(cls, aux_data, children):
    #     self = object.__new__(cls)
    #     self.params_treedef, self.params_leaves, self.graph = aux_data
    #     (self.unconditional,) = children
    #     return self

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
        autovmap: bool = False,
    ):
        """Reconstructs the module from different parameter representations.

        This method dispatches to the correct reconstruction logic based on the
        input type.

        If autovmap is True, an object is returned that behaves almost like
        the module except that function calls are automatically vectorized
        (via vmap) over parameters and inputs.

        Args:
            params: Can be a single array, a list of arrays, a dict, or a
                full nnx state.
            autovmap: If True, wrap the reconstruction in an AutoVmapReconstructor.
        """
        if autovmap:
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


# jax.tree_util.register_pytree_node(
#     ModuleReconstructor,
#     ModuleReconstructor._tree_flatten,
#     ModuleReconstructor._tree_unflatten,
# )


@dataclass(frozen=True)
class AutoVmapReconstructor(nnx.Pytree):
    r"""Automatic vectorization for module reconstruction with batched parameters.

    This class provides a solution for bijections that do not natively support
    batching over parameters, but can also be used generally.
    It wraps :class:`ModuleReconstructor` and automatically applies ``jax.vmap``
    to function calls, enabling parameter batching for any bijection.

    The wrapper intercepts function calls and applies vectorization over the parameter
    batch dimensions while handling input rank specifications correctly. This is
    particularly useful for coupling layers where different parameter sets need to
    be applied to different elements of the input.

    Key features:
        - Transparent function call vectorization via ``jax.vmap``
        - Automatic parameter batch dimension handling
        - Dynamic function signature modification to include input rank specifications
        - Support for both callable attributes and direct method calls

    Warning:
        Parameters inside modules must use explicit ``.value`` access as implicit
        parameter access can cause vmap errors with NNX Param types.
        Keyword arguments cannot be vectorized over - only positional arguments
        batching via ``input_ranks`` specification is supported.

    Example:
        >>> batch_size, event_size = 2, 4
        >>> # Bijection that doesn't support parameter batching
        >>> template = bijx.ModuleReconstructor(SomeBijection())
        >>> batched_params = jnp.zeros((batch_size, template.params_total_size))
        >>>
        >>> # Create auto-vmapped version
        >>> auto_bij = template.from_params(batched_params, autovmap=True)
        >>>
        >>> # Function calls are automatically vectorized
        >>> x = jnp.ones((batch_size, event_size))
        >>> ld = jnp.zeros((batch_size,))
        >>> y, log_det = auto_bij.forward(x, ld, input_ranks=(1, 0))
    """

    reconstructor: ModuleReconstructor
    params: nnx.Data[nnx.State | dict | list[jax.Array] | jax.Array]
    params_rank: nnx.Data[int | dict] = 1

    def __call__(self, fn_name, *args, input_ranks: tuple[int, ...] = (0, 0), **kwargs):

        input_ranks = tuple(input_ranks)
        input_ranks += (None,) * (len(args) - len(input_ranks))

        @autovmap(
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
    r"""General coupling layer with flexible masking and bijection support.

    Implements the fundamental coupling layer transformation where input is split
    into active and passive components, with the passive part conditioning the
    transformation applied to the active part. This maintains invertibility while
    enabling complex parameter-dependent transformations.

    Key features:
        - Flexible masking via :class:`BinaryMask` with split or multiplicative modes
        - Automatic parameter management through :class:`ModuleReconstructor`
        - Support for arbitrary bijections with automatic vectorization
        - Configurable event rank for scalar vs. (e.g.) vector bijections
        - Proper log-density computation with broadcasting/summation

    Args:
        embedding_net: Neural network that maps passive components to bijection
            parameters. Must output parameters compatible with bijection_reconstructor.
        mask: Binary mask defining active/passive split pattern.
        bijection_reconstructor: Template for reconstructing parameterized bijections.
        bijection_event_rank: Event rank of the underlying bijection
            (0 for scalar, 1 for vector).
        split: If True, use indexing-based masking;
            if False, use multiplicative masking.

    Note:
        When using multiplicative masking (split=False), log-density changes are
        automatically masked to exclude passive components from density computation.
        The embedding network output shape must match the total parameter size
        required by the bijection reconstructor, not just the active part.

    Example:
        >>> # Create mask and bijection template
        >>> mask = bijx.checker_mask((4,), parity=True)
        >>> spline = bijx.MonotoneRQSpline(10, (), rngs=rngs)
        >>> spline_template = bijx.ModuleReconstructor(spline)
        >>>
        >>> # Network producing parameters for active components
        >>> param_net = bijx.nn.nets.MLP(
        ...     in_features=mask.count_secondary,
        ...     out_features=mask.count_primary * spline_template.params_total_size,
        ...     rngs=rngs
        ... )
        >>>
        >>> # Reshape to coupling layer parameter shape
        >>> param_reshape = lambda p: p.reshape(*p.shape[:-1], mask.count_primary, -1)
        >>>
        >>> # Create coupling layer
        >>> layer = bijx.GeneralCouplingLayer(
        ...     nnx.Sequential(param_net, param_reshape),
        ...     mask, spline_template, bijection_event_rank=0,
        ... )
        >>>
        >>> batch_size = 3
        >>> x = jnp.ones((batch_size, 4))
        >>> y, log_det = layer.forward(x, jnp.zeros((batch_size,)))
    """

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
        self.bijection_reconstructor = nnx.data(bijection_reconstructor)
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
        bijection = self.bijection_reconstructor.from_params(params, autovmap=True)

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
