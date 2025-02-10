from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def cyclic_corr(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        d-dimensional array.
    """
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.mean(arr1 * shifted)

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@jax.jit
def cyclic_tensor(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x, y] = arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        2*d-dimensional array."""
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, arr1 * shifted

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@partial(jax.jit)
def cyclic_corr_mat(arr: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr[x,x+y]``.

    x and y are d-dimensional (lattice) indices.
    `arr` is a 2*d dimensional array.
    The sum is executed with periodic boundary conditions.

    This function is related to `cyclic_tensor` and `cyclic_corr`:
        >>> a, b = jnp.ones((2, 12, 12))
        >>> c1 = cyclic_corr(a, b)
        >>> c2 = jnp.mean(cyclic_tensor(a, b), 0)
        >>> jnp.all(c1 == c2).item()
        True
        >>> outer_product = jnp.einsum('ij,kl->ijkl', a, b)
        >>> c3 = cyclic_corr_mat(outer_product)
        >>> jnp.all(c2 == c3).item()
        True

    Args:
        arr: 2*d-dimensional array. x is the index of the first d
            dimensions, y is the index of the last d dimensions.

    Returns:
        d-dimensional array.
    """
    dim = arr.ndim // 2
    shape = arr.shape[:dim]
    assert shape == arr.shape[dim:], 'Invalid outer_product shape.'
    lattice_size = np.prod(shape)
    arr = arr.reshape((lattice_size,) * 2)

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.trace(arr[:, shifted.flatten()])

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    idx = jnp.arange(lattice_size).reshape(shape)
    _, c = fn(idx)
    return c.reshape(shape) / lattice_size
