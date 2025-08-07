"""Scalar field theory utilities.

This module provides computational tools for scalar field theory on discrete
lattices with periodic boundary conditions. It implements correlation function
estimators, action terms, and field observables commonly used in Monte Carlo
simulations of lattice field theories.

In some most places periodic boundary conditions are assumed.
"""

from functools import partial, reduce

import chex
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def cyclic_corr(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    r"""Compute cyclic correlation function with periodic boundary conditions.

    Computes $C(x) = \frac{1}{N} \sum_y f_1(y) f_2(y+x)$ where the sum
    over $y$ uses periodic boundary conditions and $N$ is the total number
    of lattice sites.

    This is a building block for computing two-point correlation
    functions in lattice field theory, exploiting translational invariance
    to improve statistical estimates.

    Args:
        arr1: First field configuration array of shape $(L_1, \ldots, L_d)$.
        arr2: Second field configuration array of shape $(L_1, \ldots, L_d)$.

    Returns:
        Correlation function array of shape $(L_1, \ldots, L_d)$ where
        element at position $x$ gives $C(x)$.

    Example:
        >>> phi = jnp.ones((4, 4))  # Constant field
        >>> corr = cyclic_corr(phi, phi)
        >>> jnp.allclose(corr, 1.0)  # Should be constant 1
        Array(True, dtype=bool)
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
    r"""Compute full correlation tensor without averaging over positions.

    Computes $T(x,y) = f_1(y) f_2(y+x)$ for all lattice positions $x$ and $y$,
    using periodic boundary conditions for the shift $y+x$.

    This provides the raw correlation data before averaging over the translation
    group. The cyclic correlation :func:`cyclic_corr` is obtained by averaging
    this tensor over the $y$ index.

    Args:
        arr1: First field configuration array of shape $(L_1, \ldots, L_d)$.
        arr2: Second field configuration array of shape $(L_1, \ldots, L_d)$.

    Returns:
        Correlation tensor of shape $(L_1, \ldots, L_d, L_1, \ldots, L_d)$
        where the first $d$ indices correspond to shift $x$ and the last
        $d$ indices correspond to position $y$.

    Note:
        For large lattices, prefer :func:`cyclic_corr` when only
        the averaged correlation is needed, as that method is less
        memory intensive.
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
        return None, arr1 * shifted

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@partial(jax.jit)
def cyclic_corr_mat(arr: jnp.ndarray) -> jnp.ndarray:
    r"""Compute cyclic correlation from precomputed correlation tensor.

    Given a correlation tensor $T(x,y)$, computes the cyclic correlation
    $C(x) = \frac{1}{N} \sum_y T(x, x+y)$ using periodic boundary conditions.

    This is equivalent to :func:`cyclic_corr` but operates on precomputed
    tensor data rather than field configurations. Useful when the correlation
    tensor has been computed via other means (e.g., outer products).

    Args:
        arr: Correlation tensor of shape $(L_1, \ldots, L_d, L_1, \ldots, L_d)$
            where the first $d$ dimensions index shift $x$ and the last
            $d$ dimensions index position $y$.

    Returns:
        Cyclic correlation array of shape $(L_1, \ldots, L_d)$.

    Example:
        >>> a, b = jnp.ones((2, 4, 4))
        >>> # These three approaches are equivalent:
        >>> c1 = cyclic_corr(a, b)
        >>> c2 = jnp.mean(cyclic_tensor(a, b), axis=(2, 3))
        >>> outer = jnp.einsum('ij,kl->ijkl', a, b)
        >>> c3 = cyclic_corr_mat(outer)
        >>> jnp.allclose(c1, c2) and jnp.allclose(c2, c3)
        Array(True, dtype=bool)
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
    r"""Estimate two-point correlation function from field samples.

    Computes the two-point correlation function (propagator):
    $G(x) = \langle \phi(0) \phi(x) \rangle$

    where $\langle \cdot \rangle$ denotes the expectation value over the
    field distribution. Exploits translational invariance by computing
    $\frac{1}{N} \sum_y \langle \phi(y) \phi(x+y) \rangle$ to improve
    statistical accuracy.

    Args:
        phis: Monte Carlo samples of field configurations with shape
            $(\text{batch}, L_1, \ldots, L_d)$.
        average: If True, average over samples. If False, return per-sample
            correlations for further analysis.

    Returns:
        Two-point correlation function of shape $(L_1, \ldots, L_d)$ if
        ``average=True``, otherwise shape $(\text{batch}, L_1, \ldots, L_d)$.

    Example:
        >>> # Generate correlated field samples (simplified example)
        >>> phis = jnp.ones((100, 8, 8))  # 100 samples of 8x8 lattice
        >>> G = two_point(phis)
        >>> G.shape
        (8, 8)
    """
    corr = jax.vmap(cyclic_corr)(phis, phis)
    return jnp.mean(corr, axis=0) if average else corr


@jax.jit
def two_point_central(phis: jnp.ndarray) -> jnp.ndarray:
    r"""Estimate connected two-point correlation function.

    Computes the connected (central) two-point function:

    $$
    G_c(x) =
    \langle \phi(0) \phi(x) \rangle
    - \langle \phi(0) \rangle \langle \phi(x) \rangle
    $$

    Args:
        phis: Monte Carlo samples of field configurations with shape
            $(\text{batch}, L_1, \ldots, L_d)$.

    Returns:
        Connected two-point correlation function of shape $(L_1, \ldots, L_d)$.

    Example:
        >>> # For a field with non-zero mean, connected differs from full
        >>> phis = 0.5 + 0.1 * jax.random.normal(key, (100, 8, 8))
        >>> G_full = two_point(phis)
        >>> G_conn = two_point_central(phis)
        >>> jnp.max(jnp.abs(G_full - G_conn)) > 0.1  # Significant difference
        Array(True, dtype=bool)
    """
    phis_mean = jnp.mean(phis, axis=0)
    outer = phis_mean * jnp.mean(phis_mean)

    return two_point(phis, True) - outer


@jax.jit
def correlation_length(two_point: jax.Array) -> jax.Array:
    r"""Extract correlation length from two-point function.

    Estimates the correlation length $\xi$ by fitting the asymptotic
    exponential decay of the correlation function. Uses the effective
    mass estimator:
    $m_{\text{eff}}(t) = \text{arccosh}\left(\frac{G(t-1) + G(t+1)}{2G(t)}\right)$

    The correlation length is $\xi = 1/m_{\text{eff}}$ averaged over
    suitable time slices.

    This method assumes the correlator has the asymptotic form
    $G(x) \sim e^{-\abs{x}/\xi}$ for large separations.

    Args:
        two_point: Connected two-point correlation function, typically
            from :func:`two_point_central`. Should have shape compatible
            with marginalizing over spatial directions.

    Returns:
        Scalar estimate of the correlation length $\xi$.
    """
    marginal = jnp.mean(two_point, axis=0)
    arg = (jnp.roll(marginal, 1) + jnp.roll(marginal, -1)) / (2 * marginal)
    mp = jnp.arccosh(arg[1:])
    return 1 / jnp.nanmean(mp)


@jax.jit
def kinetic_term(phi: jax.Array) -> jax.Array:
    r"""Compute kinetic energy density for scalar field theory.

    Computes the discrete kinetic energy term:
    $T(x) = \sum_{\mu=1}^d [\phi(x+\hat{\mu}) - \phi(x)]^2$

    where $\hat{\mu}$ are the unit vectors in each lattice direction.
    This approximates the continuum kinetic term $(\nabla \phi)^2$ using
    finite differences with periodic boundary conditions.

    Args:
        phi: Scalar field configuration of shape $(L_1, \ldots, L_d)$.

    Returns:
        Kinetic energy density at each lattice site, same shape as input.

    Example:
        >>> # Constant field has zero kinetic energy
        >>> phi = jnp.ones((4, 4))
        >>> T = kinetic_term(phi)
        >>> jnp.allclose(T, 0.0)
        Array(True, dtype=bool)
    """
    a = reduce(jnp.add, [(jnp.roll(phi, 1, y) - phi) ** 2 for y in range(phi.ndim)])
    return a


@partial(jax.jit, static_argnums=(2,))
def poly_term(phi: jax.Array, coeffs: jax.Array, even: bool = False) -> jax.Array:
    r"""Compute polynomial potential energy density.

    Evaluates a polynomial potential $V(\phi) = \sum_{n=0}^N c_n \phi^n$
    at each lattice site. Supports both general polynomials and even
    polynomials in $\phi^2$.

    Args:
        phi: Scalar field configuration of shape $(L_1, \ldots, L_d)$.
        coeffs: Polynomial coefficients $[c_N, c_{N-1}, \ldots, c_1, c_0]$
            in descending order of powers (NumPy convention).
        even: If True, evaluate polynomial in $\phi^2$ instead of $\phi$,
            giving $V(\phi) = \sum_{n=0}^N c_n (\phi^2)^n$.

    Returns:
        Potential energy density at each lattice site, same shape as input.

    Example:
        >>> phi = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> # Quadratic potential: V(phi) = phi^2
        >>> coeffs = jnp.array([1.0, 0.0])  # [c_1, c_0] = [1, 0]
        >>> V = poly_term(phi, coeffs)
        >>> jnp.allclose(V, phi**2)
        Array(True, dtype=bool)
    """
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
    r"""Compute action density for $\phi^4$ scalar field theory.

    Implements the standard $\phi^4$ action density:
    $\mathcal{L}(x) = (\nabla \phi)^2 + m^2 \phi^2 + \lambda \phi^4$

    This combines the kinetic energy, mass term, and quartic self-interaction.
    The resulting action is $S = \sum_x \mathcal{L}(x)$.

    Args:
        phi: Scalar field configuration of shape $(L_1, \ldots, L_d)$.
        m2: Bare mass squared parameter $m^2$.
        lam: Quartic coupling constant $\lambda$. If None, omits the
            $\phi^4$ interaction term.

    Returns:
        Action density at each lattice site, same shape as input.

    Note:
        Does not include common overall normalization factor like $1/2$.

    Example:
        >>> phi = jnp.ones((4, 4))  # Constant field
        >>> action_density = phi4_term(phi, m2=1.0, lam=0.1)
        >>> # For constant field: kinetic=0, mass=m2, interaction=lam
        >>> expected = 1.0 + 0.1  # m2 * 1^2 + lam * 1^4
        >>> jnp.allclose(action_density, expected)
        Array(True, dtype=bool)
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
    r"""Alternative parameterization of $\phi^4$ action using hopping parameter.

    $$
    \mathcal{L}(x) =
    -2\kappa \phi(x) \sum_{\mu} \phi(x+\hat{\mu})
    + (1-2\lambda) \phi(x)^2 + \lambda \phi(x)^4
    $$

    Args:
        phi: Scalar field configuration of shape $(L_1, \ldots, L_d)$.
        kappa: Hopping parameter controlling kinetic term strength.
        lam: Self-interaction parameter. If None, uses $\lambda = 0$.

    Returns:
        Action density at each lattice site, same shape as input.
    """
    kinetic = (
        (-2 * kappa)
        * phi
        * reduce(jnp.add, [jnp.roll(phi, 1, y) for y in range(phi.ndim)])
    )
    if lam is None:
        return kinetic + phi**2
    return kinetic + (1 - 2 * lam) * phi**2 + lam * phi**4
