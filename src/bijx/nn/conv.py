r"""
Symmetric convolutions with group invariance for lattice field theory.

This module implements convolutions that preserve spatial symmetries,
particularly useful for lattice field theory applications.
This is implemented by explicit parameter sharing forcing equivariance.
In particular, only the "trivial" representation is supported with respect to
the channel "fibers" (i.e. spatial scalar "images" to scalar "images").

Key components:
    - :class:`ConvSym`: Symmetric convolution layer with orbit-based parameter sharing
    - :func:`kernel_d4`: D4 symmetry group (rotations and reflections) for 2D lattices
    - :func:`kernel_equidist`: Distance-based parameter sharing for isotropic kernels.

Symmetric convolutions implement convolutions that commute with group actions:
$$g \cdot (W * x) = W * (g \cdot x)$$
where $g$ is a group element (rotation, reflection, etc.)
and $W$ is the convolution kernel.

This is achieved through orbit decomposition, where lattice sites are grouped
into orbits of equivalent positions under the symmetry group. Parameters are
shared within each orbit, reducing the parameter count and enforcing
the desired symmetries.

The orbit construction is implemented somewhat naively, by simply applying
the group operators to lattice indices and collecting the results.
"""

import typing as tp
from functools import partial

import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn import dtypes


def conv_indices(shape: tuple[int, ...], return_flat=True, center=True):
    """Generate index matrix for translation invariant convolution layers.

    Creates the orbital index structure for translation-invariant convolutions
    by computing relative position differences between all pairs of lattice sites.
    This forms the basis for implementing convolutions through orbit decomposition.

    Args:
        shape: Spatial dimensions of the lattice.
        return_flat: If True, flatten spatial dimensions to single axis.
        center: If True, center indices around lattice midpoint.

    Returns:
        Index array with shape ``[spatial_rank, *shape, *shape]`` if
        ``return_flat=False``, or ``[spatial_rank, prod(shape), prod(shape)]``
        if ``return_flat=True``.
        Entry ``[i, x, y]`` contains the i-th component of displacement ``(y-x)``.
    """
    xs_grid = np.indices(shape)
    flat = np.reshape(xs_grid, (len(shape), np.prod(shape)))
    mods = np.array(shape).reshape((-1, 1, 1))
    shift = np.array(shape).reshape((-1, 1, 1)) // 2 if center else 0
    added = np.mod(flat[:, None, :] - flat[:, :, None] + shift, mods)
    full_shape = (len(shape), *shape, *shape)
    return added if return_flat else added.reshape(full_shape)


def _lattice_distances(shape: tuple[int, ...]):
    """Compute squared distances from lattice sites to central origin.

    Calculates the Euclidean distance squared from each lattice site to the
    central origin, which is positioned at the lattice midpoint. This is used
    for creating distance-based orbit decompositions and isotropic kernels.

    Args:
        shape: Spatial dimensions of the lattice.

    Returns:
        Integer array of given shape where each entry contains the squared
        distance from that site to the lattice center at position L//2
        in each dimension of length L.
    """
    coords = [np.arange(-((s - 1) // 2), 1 + s // 2) for s in shape]
    coords = np.meshgrid(*coords, indexing="ij")
    dist = sum(c**2 for c in coords)
    return dist


def _gather_orbit_indices(orbits: jnp.ndarray):
    """Renumber orbit indices to be contiguous starting from zero.

    Takes an array of orbit indices (which may have gaps or non-sequential
    values) and remaps them to a contiguous sequence 0, 1, 2, ..., preserving
    the equivalence classes.

    Args:
        orbits: Array of orbit indices (possibly non-contiguous).

    Returns:
        Tuple of (num_orbits, renumbered_orbits) where num_orbits is the
        total number of distinct orbits and renumbered_orbits has the same
        shape as input but with contiguous indices.
    """
    unique = np.unique(orbits)
    index_map = np.empty(unique[-1] + 1, dtype=int)
    index_map[unique] = np.arange(len(unique))
    orbits = index_map[orbits]
    return len(unique), orbits


def unique_index_kernel(shape: tuple[int, ...]):
    """Generate unique indices for lattice sites ordered by distance to center.

    Creates a unique identifier for each lattice site, with indices assigned
    in order of increasing distance from the lattice center. Sites at the same
    distance receive consecutive indices, which facilitates orbit decomposition
    algorithms.

    Args:
        shape: Spatial dimensions of the lattice.

    Returns:
        Integer array of given shape where each entry contains a unique index.
        Indices are assigned in order of increasing distance from the center,
        with the central site receiving index 0.
    """
    dist = _lattice_distances(shape)
    index = np.empty(np.prod(shape), dtype=int)
    index[np.argsort(dist.flatten())] = np.arange(np.prod(shape))
    index = index.reshape(shape)
    return index


def flip_lattice(lattice: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Apply reflection symmetry along specified axis with proper boundary handling.

    Implements lattice reflection that respects periodic boundary conditions.
    The reflection is followed by a boundary-preserving roll operation to
    maintain correct periodicity at lattice edges.

    Args:
        lattice: Input lattice array to reflect.
        axis: Spatial axis along which to apply the reflection.

    Returns:
        Reflected lattice array with same shape as input.

    Note:
        The roll operation by (shape[axis] % 2 - 1) ensures that the reflection
        operation is consistent with periodic boundary conditions, which is
        crucial for maintaining proper lattice symmetries.
    """
    return np.roll(np.flip(lattice, axis), lattice.shape[axis] % 2 - 1, axis)


def rot_lattice_90(lattice: jnp.ndarray, ax1: int, ax2: int) -> jnp.ndarray:
    """Apply 90-degree rotation in the plane defined by two axes.

    Implements a 90-degree counterclockwise rotation by swapping the specified
    axes and applying a reflection. This operation is a fundamental building
    block for constructing the D4 symmetry group on 2D lattices.

    Args:
        lattice: Input lattice array to rotate.
        ax1: First axis defining the rotation plane.
        ax2: Second axis defining the rotation plane.

    Returns:
        Rotated lattice array with same shape as input.

    Note:
        The composition of axis swapping followed by reflection along ax1
        implements a proper 90-degree rotation that preserves the lattice
        structure and boundary conditions.
    """
    lattice = np.swapaxes(lattice, ax1, ax2)
    lattice = flip_lattice(lattice, ax1)
    return lattice


def gather_orbits(
    shape: tuple[int, ...],
    transformations: list[tp.Callable[[jnp.ndarray], jnp.ndarray]],
) -> tuple[int, jnp.ndarray]:
    """Compute orbit decomposition for lattice under given symmetry group.

    Determines equivalence classes (orbits) of lattice sites under the action
    of the symmetry group generated by the provided transformations. Sites in
    the same orbit can be transformed into each other by group operations,
    enabling parameter sharing in symmetric convolutions.

    Args:
        shape: Spatial dimensions of the lattice.
        transformations: List of group generators (symmetry operations).
            Each transformation takes a lattice array and returns the
            transformed array under that group element.

    Returns:
        Tuple containing:
            - num_orbits: Total number of distinct orbits
            - orbit_indices: Array of same shape as lattice, where each entry
              contains the orbit index (0 to num_orbits-1) for that site
    """
    # start by assigning a unique 'orbit' to each index of the lattice
    lattice = unique_index_kernel(shape)
    unique_indices = lattice.flatten()

    # generate the stack of transformed lattices
    partial_orbits = np.empty(
        (len(unique_indices), len(transformations) + 1), dtype=int
    )
    partial_orbits[:, 0] = unique_indices
    for i, op in enumerate(transformations):
        partial_orbits[:, i + 1] = op(lattice).flatten()

    # maps orbit -> set(indices)
    orbit_members = dict()
    # maps index -> orbit
    orbit_index = np.full_like(unique_indices, -1)

    # consider all indices that appear at a given site given the
    # 'generator' transformations applied above
    for parts in partial_orbits:
        orbits = set()  # track what orbits need to be merged
        update_indices = set()  # track for which indices orbit needs updating
        for i in parts:
            if orbit_index[i] != -1:  # index was already assigned to an orbit
                orbits.add(orbit_index[i])
            else:
                update_indices.add(i)

        # let the orbit index be the minimum of the site indices it contains
        new_orbit = np.min(parts)
        if len(orbits) != 0:
            new_orbit = min(new_orbit, min(orbits))

        if new_orbit not in orbits:
            members = update_indices.copy()
            orbit_members[new_orbit] = members
        else:
            members = orbit_members[new_orbit]
            members.update(update_indices)

        # merge orbits
        for orbit in orbits:
            if orbit == new_orbit:
                continue
            new_members = orbit_members[orbit]
            members.update(new_members)
            update_indices.update(new_members)
            del orbit_members[orbit]

        for i in update_indices:
            orbit_index[i] = new_orbit

    new_orbits = orbit_index[unique_indices].reshape(lattice.shape)
    return _gather_orbit_indices(new_orbits)


def kernel_d4(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Compute orbit decomposition for D4 dihedral group symmetry.

    Implements the D4 symmetry group consisting of rotations and reflections
    for 2d shapes. Naturally generalizes to higher dimensions.

    Example:
        >>> # 3x3 kernel with D4 symmetry
        >>> num_orbits, orbits = kernel_d4((3, 3))
        >>> # num_orbits = 3 (center, corners, edges)

    Args:
        shape: Spatial dimensions of the lattice (must be square for rotations).

    Returns:
        Tuple containing:
            - num_orbits: Number of distinct orbits under D4 symmetry
            - orbit_indices: Array, same shape as lattice with orbit index for each site

    Raises:
        AssertionError: If lattice is not square (required for rotation symmetry).
    """
    assert all(
        shape[0] == li for li in shape[1:]
    ), "Rotation requires all side lengths to be equal."
    transformations = [partial(flip_lattice, axis=i) for i in range(len(shape))]
    for i in range(len(shape)):
        # probably redundantly many
        for j in range(i, len(shape)):
            transformations.append(partial(rot_lattice_90, ax1=i, ax2=j))

    return gather_orbits(shape, transformations)


def kernel_equidist(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Compute orbit decomposition based on distance from lattice center.

    Creates orbits by grouping lattice sites at equal Euclidean distance
    from the central origin.

    Args:
        shape: Spatial dimensions of the lattice.

    Returns:
        Tuple containing:
            - num_orbits: Number of distinct distance shells
            - orbit_indices: Array, same shape as lattice with orbit index for each site

    Example:
        >>> # 5x5 kernel with distance-based orbits
        >>> num_orbits, orbits = kernel_equidist((5, 5))
        >>> # Sites at distances 0, 1, √2, 2, √5, etc. form separate orbits
    """
    dist = _lattice_distances(shape)
    return _gather_orbit_indices(dist)


def unfold_kernel(kernel_params: jnp.ndarray, orbits: jnp.ndarray) -> jnp.ndarray:
    """Expand symmetric kernel parameters to full convolution kernel.

    Reconstructs the complete convolution kernel from the compressed orbit
    representation by broadcasting shared parameters to all sites within
    each orbit. This operation converts from the efficient symmetric
    representation back to standard convolution kernel format.

    Note:
        This function is the inverse of :func:`fold_kernel`.

    Args:
        kernel_params: Compressed parameters with shape
            (num_orbits, in_channels, out_channels).
        orbits: Integer array giving the orbit index for each lattice site.

    Returns:
        Full convolution kernel with shape
        ``(*lattice_shape, in_channels, out_channels)``
        where ``lattice_shape`` matches the shape of the orbits array.
    """
    return kernel_params[orbits]


def resize_kernel_weights(
    kernel: jnp.ndarray,
    new_shape: int | tuple[int, ...],
    *,
    mode: str = "constant",
    constant_values: float = 0.0,
    **pad_args,
) -> jnp.ndarray:
    """Resize convolution kernel while preserving periodicity and symmetries.

    Increases the spatial size of a convolution kernel through careful padding
    that respects periodic boundary conditions and maintains proper normalization.
    This is particularly important for symmetric kernels with boundary effects.

    Note:
        For even-sized dimensions, the boundary values are handled specially
        to maintain proper circular padding. Values at wrap-around edges are
        split between copies to preserve total weight normalization.

    Example:
        >>> # Resize 3x3 kernel to 5x5
        >>> kernel_3x3 = jnp.ones((3, 3, 1, 1))
        >>> kernel_5x5 = resize_kernel_weights(kernel_3x3, (5, 5))

    Args:
        kernel: Convolution kernel to resize with shape
            ``(*spatial_dims, in_channels, out_channels)``.
        new_shape: Target spatial dimensions. Can be integer (for square kernel)
            or tuple specifying each dimension.
        mode: Padding mode for dimensions beyond the original kernel size.
        constant_values: Fill value when using constant padding mode.
        **pad_args: Additional arguments passed to numpy padding function.

    Returns:
        Resized kernel with spatial shape matching new_shape while preserving
        the original channel dimensions.
    """
    shape = kernel.shape[:-2]
    if isinstance(new_shape, tuple):
        assert len(new_shape) == len(
            shape
        ), "The dimension of the new shape does not match existing kernel."
    else:
        new_shape = (new_shape,) * len(shape)

    # in even dimensions, copy the 'wrap-around' indices (at the edge)
    wraps = [
        (1, 0) if (old % 2 == 0) and new > old else (0, 0)
        for new, old in zip(new_shape, shape)
    ]

    kernel = np.pad(kernel, [*wraps, (0, 0), (0, 0)], "wrap")

    # divide copied values by two
    _slice = np.index_exp[:]
    for dim, length in enumerate(shape):
        if length % 2 == 0:
            kernel[_slice * dim + ([0, -1],)] /= 2

    # add zeros to reach desired shape
    padding = [
        ((new - old - 1) // 2, (old % 2) + (new - old) // 2) if new > old else (0, 0)
        for new, old in zip(new_shape, shape)
    ]
    w = np.pad(
        kernel,
        padding + [(0, 0)] * 2,
        mode,
        constant_values=constant_values,
        **pad_args,
    )

    # crop
    crop = ()
    for new, old in zip(new_shape, shape):
        if new >= old:
            crop += np.index_exp[:]
        else:
            crop += np.index_exp[
                (old - new + (old % 2)) // 2 : -(old - new + (new % 2)) // 2
            ]
    return w[crop]


def fold_kernel(
    kernel_weights: jnp.ndarray, orbits: jnp.ndarray, orbit_count: int
) -> jnp.ndarray:
    """Extract symmetric parameters from full convolution kernel.

    Compresses a full convolution kernel into the orbit-based symmetric
    representation by averaging parameters within each orbit. This is useful
    for initializing symmetric convolutions from pre-trained standard kernels.

    Note:
        This function is the inverse of :func:`unfold_kernel`.

    Args:
        kernel_weights: Full convolution kernel to compress.
        orbits: Integer array giving the orbit index for each lattice site.
        orbit_count: Total number of distinct orbits.

    Returns:
        Compressed parameter array with shape (num_orbits, in_channels, out_channels)
        containing the average parameter values for each orbit.
    """
    in_channels, out_channels = kernel_weights.shape[-2:]
    assert orbits.shape == kernel_weights.shape[:-2]

    w_raw = np.zeros((orbit_count, in_channels, out_channels))
    count = np.zeros((orbit_count, 1, 1))

    flat_kernel = kernel_weights.reshape(-1, in_channels, out_channels)
    for index, ws in zip(orbits.flatten(), flat_kernel):
        w_raw[index] += ws
        count[index] += 1

    return w_raw / count


class ConvSym(nnx.Module):
    r"""Symmetric convolution layer with orbit-based parameter sharing.

    Implements convolutions that preserve discrete symmetries by sharing parameters
    among equivalent lattice sites (orbits). This dramatically reduces parameter
    count while maintaining desired symmetries, making it particularly suitable
    for lattice field theory and physics applications.

    Note:
        The orbit_function determines the symmetry group. Common choices:
        - :func:`kernel_d4`: D4 dihedral group (rotations + reflections)
        - :func:`kernel_equidist`: Radial symmetry (distance-based orbits)
        - None: No symmetry (standard convolution)

    Example:
        >>> # D4-symmetric convolution for 2D lattice
        >>> conv = ConvSym(
        ...     in_features=1, out_features=16, kernel_size=(3, 3),
        ...     orbit_function=kernel_d4, rngs=rngs
        ... )
        >>> y = conv(phi[..., None])  # Preserves rotations and reflections

    Args:
        in_features: Number of input feature channels.
        out_features: Number of output feature channels.
        kernel_size: Spatial dimensions of convolution kernel.
        orbit_function: Function to compute orbit decomposition (default: D4 symmetry).
        strides: Convolution stride in each spatial dimension.
        padding: Padding strategy ('CIRCULAR' for periodic boundaries).
        input_dilation: Input dilation factors for each spatial dimension.
        kernel_dilation: Kernel dilation factors (atrous convolution).
        feature_group_count: Number of feature groups for grouped convolution.
        use_bias: Whether to include learnable bias terms.
        mask: Optional mask for weights during masked convolution.
        dtype: Computation dtype (inferred if None).
        param_dtype: Parameter initialization dtype.
        precision: Numerical precision specification.
        kernel_init: Kernel parameter initializer.
        bias_init: Bias parameter initializer.
        conv_general_dilated: Convolution implementation function.
        promote_dtype: Dtype promotion function.
        rngs: Random number generator state.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        orbit_function: tp.Callable | None = kernel_d4,
        strides: int | tp.Sequence[int] = 1,
        *,
        padding: ftp.PaddingLike = "CIRCULAR",
        input_dilation: int | tp.Sequence[int] = 1,
        kernel_dilation: int | tp.Sequence[int] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: jax.Array | None = None,
        dtype: ftp.Dtype | None = None,
        param_dtype: ftp.Dtype | None = None,
        precision: ftp.PrecisionLike = None,
        kernel_init: ftp.Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: ftp.Initializer = nnx.nn.linear.default_bias_init,
        conv_general_dilated: ftp.ConvGeneralDilatedT = jax.lax.conv_general_dilated,
        promote_dtype: ftp.PromoteDtypeFn = dtypes.promote_dtype,
        preferred_element_type: ftp.Dtype | None = None,
        rngs: nnx.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        else:
            kernel_size = tuple(kernel_size)

        kernel_shape = self.kernel_shape = kernel_size + (
            in_features // feature_group_count,
            out_features,
        )

        if orbit_function is not None:
            orbit_count, orbits = orbit_function(kernel_size)
            self.orbits = nnx.data(orbits)
            w_shape = (orbit_count, kernel_shape[-2], kernel_shape[-1])
        else:
            self.orbits = None
            w_shape = (np.prod(kernel_shape[:-2]), kernel_shape[-2], kernel_shape[-1])

        # Choose parameter dtype lazily to respect global precision settings
        chosen_param_dtype = param_dtype or jnp.result_type(0.0)

        kernel_key = rngs.params()
        self.kernel_params = nnx.Param(
            kernel_init(kernel_key, w_shape, chosen_param_dtype)
        )

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            bias_shape = (out_features,)
            bias_key = rngs.params()
            self.bias = nnx.Param(bias_init(bias_key, bias_shape, chosen_param_dtype))
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_dilation = input_dilation
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.mask = mask
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.conv_general_dilated = conv_general_dilated
        self.promote_dtype = promote_dtype
        self.preferred_element_type = preferred_element_type

    @property
    def kernel(self) -> nnx.Param[jax.Array]:
        """Construct full kernel from orbit-shared parameters."""
        if self.orbits is not None:
            kernel = self.kernel_params[self.orbits]
        else:
            kernel = self.kernel_params.reshape(self.kernel_shape)
        return nnx.Param(kernel)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return nnx.Conv.__call__(self, inputs)
