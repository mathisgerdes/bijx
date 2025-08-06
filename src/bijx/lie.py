r"""
Lie group operations and automatic differentiation tools.

This module provides specialized tools for working with matrix Lie groups,
particularly for applications in lattice field theory and gauge theories.
It focuses on efficient automatic differentiation with respect to group
elements, Haar measure sampling, and gradient computations on manifolds.

Key functionality:
    - Haar measure sampling for SU(N) groups
    - Automatic differentiation with respect to matrix group elements
    - Lie algebra projections and tangent space operations
    - Specialized gradient operators for group-valued functions
    - Matrix chain contractions and traces

The implementation is optimized for SU(N) groups commonly used in
lattice gauge theory, but should also generalize to other matrix groups such
as O(N).
"""

from functools import partial, wraps
from inspect import signature

import chex
import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum

from .distributions import ArrayDistribution

# -- Constants -- #

U1_GEN = 2j * jnp.ones((1, 1, 1))

SU2_GEN = 1j * jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ]
)

SU3_GEN = 1j * jnp.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1 / jnp.sqrt(3), 0, 0], [0, 1 / jnp.sqrt(3), 0], [0, 0, -2 / jnp.sqrt(3)]],
    ]
)


# -- Operations -- #


def contract(*factors, trace=False, return_einsum_indices=False):
    r"""Contract chain of matrices with Einstein summation.

    Performs matrix multiplication chain $A_1 A_2 \cdots A_n$ with automatic
    broadcasting over batch dimensions. The contraction follows left-to-right
    order with proper index management for arbitrary numbers of factors.

    Key features:
        - Handles arbitrary number of matrix factors
        - Automatic broadcasting over leading (batch) dimensions
        - Optional trace computation for closed loops
        - Can return einsum indices for debugging/inspection

    Args:
        factors: Sequence of arrays representing matrices to contract.
            Each must have at least 2 dimensions (matrix dimensions).
        trace: Whether to trace the result (connect first and last indices).
        return_einsum_indices: Whether to return einsum index strings.

    Returns:
        Contracted result, or tuple (result, in_indices, out_indices)
        if return_einsum_indices=True.

    Example:
        >>> A = jnp.ones((3, 4, 4))
        >>> B = jnp.ones((3, 4, 4))
        >>> C = contract(A, B)  # Shape (3, 4, 4)
        >>> trace_AB = contract(A, B, trace=True)  # Shape (3,)
    """
    leading = [jnp.ndim(f) - 2 for f in factors]
    assert all(
        lead >= 0 for lead in leading
    ), "all factors must be matrices (ndim >= 2)"

    indices = []
    for m, lead in enumerate(leading):
        indices.append([f"l{i}" for i in range(lead)] + [f"m{m}", f"m{m + 1}"])

    if trace:
        indices[-1][-1] = "m0"

    ind_in = ", ".join(" ".join(ind) for ind in indices)
    ind_out = " ".join(f"l{i}" for i in range(max(leading)))
    if not trace:
        ind_out += f" m0 m{len(factors)}"
    if return_einsum_indices:
        return ind_in, ind_out
    return einsum(*factors, f"{ind_in} -> {ind_out}")


def scalar_prod(a, b):
    r"""Compute scalar product between Lie algebra elements.

    Implements the standard scalar product on the Lie algebra:
    $\langle A, B \rangle = \frac{1}{2} \text{tr}(A^\dagger B)$

    Args:
        a: First Lie algebra element.
        b: Second Lie algebra element.

    Returns:
        Real scalar product value.

    Note:
        For skew-Hermitian matrices $A, B$, this gives a real result
        and defines a positive definite inner product on the Lie algebra.
    """
    return jnp.einsum("...ij,...ij", a.conj(), b) / 2


def adjoint(arr):
    r"""Compute conjugate transpose (adjoint) of matrix.

    Returns the Hermitian conjugate $A^\dagger = (A^T)^*$ of the input matrix.

    Args:
        arr: Input matrix array.

    Returns:
        Conjugate transpose of the input.
    """
    return arr.conj().swapaxes(-1, -2)


# -- Sampling -- #


@jax.vmap
def _haar_transform(z):
    # if this is a bottleneck, investigate https://github.com/google/jax/issues/8542
    q, r = jnp.linalg.qr(z)
    d = jnp.diag(r)
    d = d / jnp.abs(d)
    norm = jnp.prod(d) * jnp.linalg.det(q)
    m = jnp.einsum("ij,j->ij", q, d / norm ** (1 / len(d)))
    return m


def _sample_haar(rng, n, count):
    """Sample multiple SU(N) matrices uniformly according to Haar measure.

    Uses the QR decomposition method to generate uniformly distributed
    SU(N) matrices from Gaussian random matrices.

    Args:
        rng: Random key for sampling.
        n: Dimension of SU(N) group.
        count: Number of matrices to sample.

    Returns:
        Array of shape (count, n, n) containing SU(N) matrices.
    """
    real_part, imag_part = 1 / np.sqrt(2) * jax.random.normal(rng, (2, count, n, n))
    z = real_part + 1j * imag_part
    return _haar_transform(z)


def sample_haar(rng, n=2, batch_shape=()):
    """Sample SU(N) matrices uniformly according to Haar measure.

    Generates random SU(N) matrices distributed according to the unique
    left- and right-invariant Haar measure on the group. This is the
    standard uniform distribution for compact Lie groups.

    Args:
        rng: Random key for sampling.
        n: Dimension of SU(N) group (default: SU(2)).
        batch_shape: Shape of batch dimensions for multiple samples.

    Returns:
        SU(N) matrices of shape batch_shape + (n, n).

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> su2_matrix = sample_haar(key, n=2)  # Single SU(2) matrix
        >>> su3_batch = sample_haar(key, n=3, batch_shape=(10,))  # 10 SU(3) matrices
    """
    if batch_shape == ():
        z = _sample_haar(rng, n, 1)
        return jnp.squeeze(z, axis=0)
    size = np.prod(batch_shape)
    return _sample_haar(rng, n, size).reshape(*batch_shape, n, n)


class HaarDistribution(ArrayDistribution):
    """Distribution of SU(N) matrices under Haar measure.

    Implements the uniform distribution on the compact Lie group SU(N)
    according to the normalized Haar measure. This is the unique left-
    and right-invariant probability measure on the group.

    The distribution can handle additional base shape dimensions for
    applications like lattice gauge theory where matrices are assigned
    to lattice sites and links.

    Args:
        n: Dimension of SU(N) group.
        base_shape: Additional shape dimensions (e.g., lattice structure).
        rngs: Random number generators for sampling.

    Example:
        >>> # SU(2) matrices on a 4x4 lattice with 4 link directions
        >>> haar_dist = HaarDistribution.periodic_gauge_lattice(
        ...     n=2, lat_shape=(4, 4), rngs=rngs
        ... )
        >>> samples, _ = haar_dist.sample((100,), rng=rngs.next())
    """

    def __init__(self, n, base_shape=(), rngs=None):
        """Initialize Haar measure distribution.

        Args:
            n: Dimension of SU(N) group.
            base_shape: Additional shape dimensions before matrix dimensions.
            rngs: Random number generators.
        """
        super().__init__(event_shape=base_shape + (n, n), rngs=rngs)
        self.base_shape = base_shape
        self.n = n

    @classmethod
    def periodic_gauge_lattice(cls, n, lat_shape, rngs=None):
        """Create Haar distribution for periodic gauge lattice.

        Convenience constructor for lattice gauge theories with periodic
        boundary conditions. Creates SU(N) matrices for each lattice site
        and spatial direction.

        Args:
            n: Dimension of SU(N) gauge group.
            lat_shape: Shape of the spatial lattice.
            rngs: Random number generators.

        Returns:
            HaarDistribution with event shape lat_shape + (ndim, n, n)
            where ndim = len(lat_shape) is the number of spatial dimensions.
        """
        base_shape = (*lat_shape, len(lat_shape))
        return cls(n, base_shape, rngs)

    def sample(self, batch_shape, rng=None, **kwargs):
        """Sample SU(N) matrices from Haar measure.

        Args:
            batch_shape: Shape of batch dimensions.
            rng: Random key for sampling.
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple of (samples, log_density) where log_density is zero
            since Haar measure is the uniform distribution.
        """
        rng = self._get_rng(rng)
        samples = sample_haar(rng, self.n, batch_shape + self.base_shape)
        return samples, jnp.zeros(batch_shape)

    def log_density(self, x, **kwargs):
        """Evaluate log density under Haar measure.

        Args:
            x: SU(N) matrices to evaluate.
            **kwargs: Additional arguments (unused).

        Returns:
            Zero log density (uniform distribution on compact group).

        Note:
            The Haar measure is uniform, so the log density is constant
            (zero after normalization).
        """
        return jnp.zeros(x.shape[: -2 - len(self.base_shape)])


# -- Gradients -- #


def _isolate_argument(fun, argnum, *args, **kwargs):
    """Partially apply all but one argument of a function.

    Note: does not work on keyword arguments!
    """
    sig = signature(fun).bind(*args, **kwargs)
    sig.apply_defaults()

    args = list(sig.args)
    arg = sig.args[argnum]
    args[argnum] = None  # to be set in wrapped

    def wrapped(arg):
        args[argnum] = arg
        return fun(*args, **sig.kwargs)

    return wrapped, arg


def skew_traceless_cot(a, u):
    r"""Project cotangent vector to SU(N) Lie algebra.

    Transforms the cotangent vector from JAX's backward pass into an
    element of the SU(N) Lie algebra (traceless skew-Hermitian matrices).
    This is the natural projection for SU(N) groups.

    Mathematical operation:
        1. Compute $A^\dagger$ (conjugate transpose of cotangent)
        2. Transport to identity: $U A^\dagger$
        3. Project to skew-Hermitian: $B - B^\dagger$
        4. Project to traceless: $B - \frac{\text{tr}(B)}{n} I$

    Args:
        a: Cotangent vector from automatic differentiation.
        u: Group element at which cotangent is evaluated.

    Returns:
        Element of SU(N) Lie algebra (traceless skew-Hermitian matrix).

    Note:
        This is more efficient than explicit projection using generators
        as it avoids computing scalar products with each basis element.
    """
    # transform cotangent to tangent and
    # project to traceless skew hermitian matrices
    # -> rev jacobian to algebra element for SU(N)

    # Df^dagger
    a = jnp.swapaxes(a, -1, -2)
    # transport to identity
    a = jnp.einsum("...ij,jk->...ik", u, a)
    # project to skew symmetric
    a = jnp.swapaxes(a, -1, -2).conj() - a
    # project to traceless
    a = a - jnp.trace(a) / a.shape[-1] * np.eye(a.shape[-1])
    return a


@partial(jax.vmap, in_axes=(None, None, 0))
def _proj(a, u, gen):
    """Take scalar product with generator & multiply with it.

    This can be used to project onto a general basis of generators.
    """
    return jnp.sum(a * (gen @ u)).real * gen


def grad(fn, argnum=0, return_value=False, has_aux=False, algebra=skew_traceless_cot):
    r"""Compute gradient with respect to matrix Lie group element.

    Computes the Riemannian gradient $\nabla_g f(g)$ where $g$ is a matrix
    Lie group element and $f$ is a scalar-valued function. The gradient
    lies in the tangent space $T_g G$, which is isomorphic to the Lie algebra.

    The algebra parameter controls how the cotangent vector from automatic
    differentiation is projected to the tangent space:

    - Function (a, u) -> v: Custom projection implementation
    - Array of generators: Projection via scalar products with basis elements
    - Default: Efficient SU(N) projection without explicit generators

    Args:
        fn: Scalar-valued function to differentiate.
        argnum: Argument position of the group element.
        return_value: Whether to return function value along with gradient.
        has_aux: Whether fn returns auxiliary outputs.
        algebra: Projection method (function) or generator basis (array).

    Returns:
        Function computing gradient, or (value, gradient) if return_value=True.

    Example:
        >>> def potential(U):
        ...     return jnp.real(jnp.trace(U @ U.conj().T))
        >>> grad_potential = grad(potential)
        >>> U = sample_haar(key, n=2)
        >>> gradient = grad_potential(U)  # Element of su(2) algebra
    """
    # algebra is either an array of generators, or a function doing
    # the appropriate cotangent -> tangent projection

    # backward pass gives us Df^* (complex conjugate)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        u = args[argnum]

        if return_value:
            val, a = jax.value_and_grad(fn, argnums=argnum, has_aux=has_aux)(
                *args, **kwargs
            )
        else:
            a = jax.grad(fn, argnums=argnum, has_aux=has_aux)(*args, **kwargs)

        if callable(algebra):
            a = algebra(a, u)
        else:
            a = jnp.sum(_proj(a, u, algebra), axis=0)

        return (val, a) if return_value else a

    return wrapped


def value_grad_divergence(fn, u, gens):
    """Compute the gradient and Laplacian (i.e. divergence of grad).

    This is done using two backward passes, and using an explicit
    basis of tangent vectors at the identity (`gens`).

    The given function is assumed to give scalar outputs.
    """

    def component(u, gen):
        tang = gen @ u
        pot, jvp = jax.jvp(fn, [u], [tang])
        return jvp, pot

    @jax.vmap
    def hess_prod(gen):
        tang = gen @ u
        return jax.jvp(partial(component, gen=gen), [u], [tang], has_aux=True)

    components, hess, (val, *_) = hess_prod(gens)
    tr_hess = jnp.sum(hess)
    grad = jnp.einsum("i,ijk->jk", components, gens)
    return val, grad, tr_hess


def _local_curve(fun, gen, u, left=False):
    """Make function t -> fun(exp(t gen) u).

    Note, that this is meant to take gradients t=0;
    the forward pass is, in fact, just the identity.
    """

    @jax.custom_jvp
    def fake_expm(t):
        return u

    # define a custom backward pass
    @fake_expm.defjvp
    def fake_expm_jvp(primals, tangents):
        (t_dot,) = tangents
        tangent_out = u @ (t_dot * gen) if left else (t_dot * gen) @ u
        return u, tangent_out  # here always assume t == 0

    def curve(t):
        return fun(fake_expm(t))

    return curve


def curve_grad(fun, direction, argnum=0, has_aux=False, return_value=False, left=False):
    r"""Compute directional derivative along group geodesic.

    Computes the directional derivative:
    $\frac{d}{dt} f(\ldots, \exp(t \cdot \text{direction}) \cdot g, \ldots)\Big|_{t=0}$

    This gives the rate of change of the function along the geodesic in the
    group generated by the specified Lie algebra direction.

    Key features:
        - Supports left or right group action (exp(tX)g vs g exp(tX))
        - Uses custom JVP for efficient differentiation
        - Can return function value simultaneously

    Args:
        fun: Function to differentiate.
        direction: Lie algebra element specifying the direction.
        argnum: Position of group argument (must be positional).
        has_aux: Whether fun has auxiliary outputs.
        return_value: Whether to return function value with derivative.
        left: Whether to use left group action (default: right action).

    Returns:
        Function computing directional derivative with same signature as fun.

    Note:
        For computing full gradients, the :func:`grad` function is more
        efficient as it avoids separate computations for each direction.

    Example:
        >>> direction = 1j * jnp.array([[0, 1], [-1, 0]])  # su(2) generator
        >>> directional_grad = curve_grad(potential, direction)
        >>> derivative = directional_grad(U)
    """

    @wraps(fun)
    def grad(*args, **kwargs):
        wrapped, u = _isolate_argument(fun, argnum, *args, **kwargs)
        curve_fun = _local_curve(wrapped, direction, u, left=left)

        if return_value:
            return jax.jvp(curve_fun, (0.0,), (1.0,), has_aux=has_aux)
        else:
            return jax.jacfwd(curve_fun, has_aux=has_aux)(0.0)

    return grad


def _split(x, indices, axis):
    if isinstance(x, np.ndarray):
        return
    else:
        return x._split(indices, axis)


def _unravel_array_into_pytree(pytree, axis, arr):
    """Unravel an array into a PyTree with a given structure."""
    leaves, treedef = jax.tree.flatten(pytree)
    axis = axis % arr.ndim
    shapes = [
        arr.shape[:axis] + np.shape(leaf) + arr.shape[axis + 1 :] for leaf in leaves
    ]
    parts = _split(arr, np.cumsum([np.size(leaf) for leaf in leaves[:-1]]), axis)
    reshaped_parts = [np.reshape(x, shape) for x, shape in zip(parts, shapes)]
    return jax.tree.unflatten(treedef, reshaped_parts)


def _std_basis(pytree):
    leaves, _ = jax.tree.flatten(pytree)
    ndim = sum(map(np.size, leaves))
    dtype = jax.dtypes.result_type(*leaves)
    flat_basis = jnp.eye(ndim, dtype=dtype)
    return _unravel_array_into_pytree(pytree, 1, flat_basis)


def _jacfwd_unravel(input_pytree, arr):
    return _unravel_array_into_pytree(input_pytree, -1, arr)


def _local_curve_vec(fun, gens, us):
    """Make function t -> fun(exp(t gen) u).

    Note, that this is meant to take gradients t=0;
    the forward pass is, in fact, just the identity.
    """
    leaves = jax.tree.leaves(us)

    dim = len(gens)  # dimension of vector space
    for leaf in leaves:
        assert leaf.shape[-2:] == gens.shape[-2:], (
            f"SU(N) groups must match, expected {gens.shape[-2:]} "
            f"but got {leaf.shape[-2:]}"
        )

    @jax.custom_jvp
    def fake_expm(ts, us):
        return us

    def _contract(t, t_dot, u, u_dot):
        chex.assert_shape([t, t_dot], (*u.shape[:-2], dim))
        # possibly optimize this; most of ts_dot is 0.
        tangent_out = jnp.einsum("...e,ejk,...kl->...jl", t_dot, gens, u)
        return u_dot + tangent_out

    # define a custom backward pass
    @fake_expm.defjvp
    def fake_expm_jvp(primals, tangents):
        ts, us = primals
        us = fake_expm(ts, us)
        ts_dot, us_dot = tangents
        tangent_out = jax.tree.map(_contract, ts, ts_dot, us, us_dot)
        return us, tangent_out

    def curve(ts):
        return fun(fake_expm(ts, us))

    return curve


def path_grad(fun, gens, us):
    """Compute gradient with respect to multiple matrix group inputs.

    Computes gradients of a function with respect to each matrix input
    in the PyTree ``us``. This is useful for functions that depend on multiple
    group elements simultaneously.

    Args:
        fun: Function to differentiate.
        gens: Generator basis for the Lie algebra.
        us: PyTree of matrix group elements.

    Returns:
        Tuple of (function_value, gradient_tree) where gradient_tree
        has the same structure as us but with additional generator dimension.

    Example:
        >>> gens = SU2_GEN  # su(2) generators
        >>> U1, U2 = sample_haar(rngs(), 2, (2,))
        >>> def action(us):
        ...     U1, U2 = us
        ...     return jnp.real(jnp.trace(U1 @ U2))
        >>> value, grads = path_grad(action, gens, [U1, U2])
    """
    ts = jax.tree.map(lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us)

    ts_basis = _std_basis(ts)
    curve = _local_curve_vec(fun, gens, us)
    jvp = partial(jax.jvp, curve, (ts,))
    out, jac = jax.vmap(jvp, out_axes=(None, -1))((ts_basis,))

    jac_tree = jax.tree.map(partial(_jacfwd_unravel, ts), jac)

    return out, jac_tree


def path_grad2(fun, gens, us):
    """Compute first and second derivative with respect to (each) matrix input."""
    curve = _local_curve_vec(fun, gens, us)

    def grad_fn(ts, vec):
        out, tangents = jax.jvp(curve, (ts,), (vec,))
        return tangents, out

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, -1, -1))
    def grad2_fn(ts, vec):
        grad, tangents, out = jax.jvp(
            partial(grad_fn, vec=vec), (ts,), (vec,), has_aux=True
        )
        return out, grad, tangents

    ts = jax.tree.map(lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us)
    ts_basis = _std_basis(ts)

    out, jac, jac2 = grad2_fn(ts, ts_basis)

    jac = jax.tree.map(partial(_jacfwd_unravel, ts), jac)

    jac2 = jax.tree.map(partial(_jacfwd_unravel, ts), jac2)

    return out, jac, jac2


def path_div(fun, gens, us):
    """Compute divergence of a function.

    The function is assumed to return a vector as components with respect to
    generator basis.
    """
    curve = _local_curve_vec(fun, gens, us)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0, -1))
    def grad2_fn(ts, vec):
        out, tangents = jax.jvp(
            (lambda ts: jnp.sum(vec * curve(ts))),
            (ts,),
            (vec,),
        )
        return out, tangents

    ts = jax.tree_util.tree_map(lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us)
    ts_basis = _std_basis(ts)

    out, tangents = grad2_fn(ts, ts_basis)

    return out.reshape(-1, len(gens)), jnp.sum(tangents).real
