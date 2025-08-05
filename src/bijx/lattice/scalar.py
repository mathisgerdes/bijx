from functools import partial, reduce

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
    assert shape == arr.shape[dim:], "Invalid outer_product shape."
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


@partial(jax.jit, static_argnames=("average",))
def two_point(phis: jnp.ndarray, average: bool = True) -> jnp.ndarray:
    """Estimate ``G(x) = <phi(0) phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
    ``mean_y <phi(y) phi(x+y)>`` using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.
        average: If false, average over samples is not executed.

    Returns:
        Array of shape ``(L_1, ..., L_d)`` if ``average`` is true, otherwise
        of shape ``(batch size, L_1, ..., L_d)``.
    """
    corr = jax.vmap(cyclic_corr)(phis, phis)
    return jnp.mean(corr, axis=0) if average else corr


@jax.jit
def two_point_central(phis: jnp.ndarray) -> jnp.ndarray:
    """Estimate ``G_c(x) = <phi(0) phi(x)> - <phi(0)> <phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
    ``mean_y <phi(y) phi(x+y)> - <phi(x)> mean_y <phi(x+y)>`` using periodic
    boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.

    Returns:
        Array of shape ``(L_1, ..., L_d)``.
    """
    phis_mean = jnp.mean(phis, axis=0)
    outer = phis_mean * jnp.mean(phis_mean)

    return two_point(phis, True) - outer


@jax.jit
def correlation_length(two_point: jax.Array):
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    marginal = jnp.mean(two_point, axis=0)
    arg = (jnp.roll(marginal, 1) + jnp.roll(marginal, -1)) / (2 * marginal)
    mp = jnp.arccosh(arg[1:])
    return 1 / jnp.nanmean(mp)


@jax.jit
def kinetic_term(phi: jax.Array) -> jax.Array:
    a = reduce(jnp.add, [(jnp.roll(phi, 1, y) - phi) ** 2 for y in range(phi.ndim)])
    return a


@partial(jax.jit, static_argnums=(2,))
def poly_term(phi: jax.Array, coeffs: jax.Array, even: bool = False) -> jax.Array:
    coeffs = jnp.concatenate([coeffs, np.array([0.0])])
    if even:
        phi = phi**2
    return jnp.polyval(coeffs, phi, unroll=128)


@jax.jit
def phi4_term(
    phi: jax.Array,
    m2: float,
    lam: float | None = None,
) -> jax.Array:
    """
    phi4_term = kinetic_term(phi) + m2 * phi ** 2

    Note: does not include factor 1/2 to get S ~ m^2/2 (if desired).
    """
    phi2 = phi**2
    a = kinetic_term(phi) + m2 * phi2
    if lam is not None:
        a += lam * phi2**2
    return a


@jax.jit
def phi4_term_alt(
    phi: jax.Array,
    kappa: float,
    lam: float | None = None,
) -> jax.Array:
    kinetic = (
        (-2 * kappa)
        * phi
        * reduce(jnp.add, [jnp.roll(phi, 1, y) for y in range(phi.ndim)])
    )
    mass = (1 - 2 * lam) * phi**2
    inter = lam * phi**4
    return kinetic + mass + inter
