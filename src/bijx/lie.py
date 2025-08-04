"""
This module provides a suite of tools for numerical operations on Lie groups,
with a particular focus on automatic differentiation.
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

SU2_GEN = 1j * jnp.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
])

SU3_GEN = 1j * jnp.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / jnp.sqrt(3), 0, 0],
     [0, 1 / jnp.sqrt(3), 0],
     [0, 0, -2 / jnp.sqrt(3)]],
])


# -- Operations -- #

def contract(*factors, trace=False, return_einsum_indices=False):
    """Contrast chain of matrices.

    Except for last two axes (which are contracted), broadcast
    factors from left to right (opposite of numpy).
    """
    leading = [jnp.ndim(f) - 2 for f in factors]
    assert all(l >= 0 for l in leading), 'all factors must be matrices (ndim >= 2)'

    indices = []
    for m, l in enumerate(leading):
        indices.append([f'l{i}' for i in range(l)] + [f'm{m}', f'm{m + 1}'])

    if trace:
        indices[-1][-1] = 'm0'

    ind_in = ', '.join(' '.join(ind) for ind in indices)
    ind_out = ' '.join(f'l{i}' for i in range(max(leading)))
    if not trace:
        ind_out += f' m0 m{len(factors)}'
    if return_einsum_indices:
        return ind_in, ind_out
    return einsum(*factors, f'{ind_in} -> {ind_out}')


def scalar_prod(a, b):
    """Compute scalar product between lie algebra elements a & b."""
    return jnp.einsum('...ij,...ij', a.conj(), b) / 2


def adjoint(arr):
    return arr.conj().swapaxes(-1, -2)


# -- Sampling -- #

@jax.vmap
def _haar_transform(z):
    # if this is a bottleneck, investigate https://github.com/google/jax/issues/8542
    q, r = jnp.linalg.qr(z)
    d = jnp.diag(r)
    d = d / jnp.abs(d)
    norm = jnp.prod(d) * jnp.linalg.det(q)
    m = jnp.einsum('ij,j->ij', q, d / norm**(1/len(d)))
    return m


def _sample_haar(rng, n, count):
    """Sample SU(N) matrices uniformly according to Haar measure."""
    real_part, imag_part = 1 / np.sqrt(2) * jax.random.normal(rng, (2, count, n, n))
    z = real_part + 1j * imag_part
    return _haar_transform(z)


def sample_haar(rng, n=2, batch_shape=()):
    if batch_shape == ():
        z = _sample_haar(rng, n, 1)
        return jnp.squeeze(z, axis=0)
    size = np.prod(batch_shape)
    return _sample_haar(rng, n, size).reshape(*batch_shape, n, n)


class HaarDistribution(ArrayDistribution):
    def __init__(self, n, base_shape=(), rngs=None):
        super().__init__(event_shape=base_shape + (n, n), rngs=rngs)
        self.base_shape = base_shape
        self.n = n

    @classmethod
    def periodic_gauge_lattice(cls, n, lat_shape, rngs=None):
        base_shape = (*lat_shape, len(lat_shape))
        return cls(n, base_shape, rngs)

    def sample(self, batch_shape, rng=None, **kwargs):
        rng = self._get_rng(rng)
        samples = sample_haar(rng, self.n, batch_shape + self.base_shape)
        return samples, jnp.zeros(batch_shape)

    def log_density(self, x, **kwargs):
        return jnp.zeros(x.shape[:-2-len(self.base_shape)])


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
    """Project C^(nxn) cotangent to traceless skew hermitian & take dual.

    In other words, transforms the gradient form jax's backward pass
    to an element of the Lie algebra for elements of SU(N).
    This avoids taking a scalar product for each generator, which would be
    a more general (and the default if generators are provided) method.
    See implementation of grad.
    """
    # transform cotangent to tangent and
    # project to traceless skew hermitian matrices
    # -> rev jacobian to algebra element for SU(N)

    # Df^dagger
    a = jnp.swapaxes(a, -1, -2)
    # transport to identity
    a = jnp.einsum('...ij,jk->...ik', u, a)
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
    """Compute gradient of a function with respect to matrix group element.

    The argument `algebra` determines how the general back-pass gradient
    is projected onto the tangent space.

    If `algebra` is a function (a, u) -> v, a manual projection can be
    implemented. By default, a manual projection to SU(N) is performed with
    the output tangent transported to the identity (i.e. lie algebra element).

    If `algebra` is an array, it is assumed to be a collection of basis
    elements (under the scalar product tr(a^dagger b)/2) of the tangent space
    at the identity. A projection is then performed by transporting the
    back-pass gradient to the identity, and taking a scalar product with each
    basis element to obtain components in this basis, and then constructing
    a vector from these. Note that the implementation assumes the matrix group
    is a subgroup of U(N).
    """
    # algebra is either an array of generators, or a function doing
    # the appropriate cotangent -> tangent projection

    # backward pass gives us Df^* (complex conjugate)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        u = args[argnum]

        if return_value:
            val, a = jax.value_and_grad(fn, argnums=argnum, has_aux=has_aux)(*args, **kwargs)
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
    grad = jnp.einsum('i,ijk->jk', components, gens)
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
        t_dot, = tangents
        if left:
            tangent_out = u @ (t_dot * gen)
        else:
            tangent_out = (t_dot * gen) @ u
        return u, tangent_out  # here always assume t == 0

    def curve(t):
        return fun(fake_expm(t))

    return curve


def curve_grad(fun, direction, argnum=0, has_aux=False, return_value=False, left=False):
    """Take directional derivative of function.

    This is d/dt f(..., exp(t direction) u, ...) | t = 0.

    Note that using this to compute the full gradient (i.e. to obtain the
    components in the basis of generators, then multiply with generators to
    obtain a vector) is less efficient than the implementation in grad.
    The reason is that grad avoids doing one back-pass for each generator.
    Using grad with manual projection is even faster as it avoids taking
    scalar product with the set of basis elements altogether (either implicitly
    as done here via curve gradients, or explicitly as in grad if specifying
    as set of generators).

    Args:
        fun: Any function with lie-group input at given argnum.
        direction: Lie algebra value specifying direction.
        argnum: Position of lie-group input with respect to which gradient
            is taken. Note, this must be a positional argument.
        has_aux: Whether fun has auxiliary outputs.

    Returns:
        Function with the same signature as fun.
        The output is a single value or tuple
        (value `if return_value`, gradient, aux_value `if has_aux`).
    """

    @wraps(fun)
    def grad(*args, **kwargs):
        wrapped, u = _isolate_argument(fun, argnum, *args, **kwargs)
        curve_fun = _local_curve(wrapped, direction, u, left=left)

        if return_value:
            return jax.jvp(curve_fun, (0.,), (1.,), has_aux=has_aux)
        else:
            return jax.jacfwd(curve_fun, has_aux=has_aux)(0.)

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
    shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1:] for l in leaves]
    parts = _split(arr, np.cumsum([np.size(l) for l in leaves[:-1]]), axis)
    reshaped_parts = [
        np.reshape(x, shape) for x, shape in zip(parts, shapes)]
    return jax.tree.unflatten(treedef, reshaped_parts)


def _std_basis(pytree):
    leaves, _ = jax.tree.flatten(pytree)
    ndim = sum(map(np.size, leaves))
    dtype = jax.dtypes.result_type(*leaves)
    flat_basis = jnp.eye(ndim, dtype=dtype)
    return _unravel_array_into_pytree(pytree, 1, flat_basis)


def _jacfwd_unravel(input_pytree, arr):
    return _unravel_array_into_pytree(
        input_pytree, -1, arr)


def _local_curve_vec(fun, gens, us):
    """Make function t -> fun(exp(t gen) u).

    Note, that this is meant to take gradients t=0;
    the forward pass is, in fact, just the identity.
    """
    leaves = jax.tree.leaves(us)

    dim = len(gens)  # dimension of vector space
    for leaf in leaves:
        assert leaf.shape[-2:] == gens.shape[-2:], \
            f'SU(N) groups must match, expected {gens.shape[-2:]} but got {leaf.shape[-2:]}'

    @jax.custom_jvp
    def fake_expm(ts, us):
        return us

    def _contract(t, t_dot, u, u_dot):
        chex.assert_shape([t, t_dot], (*u.shape[:-2], dim))
        # possibly optimize this; most of ts_dot is 0.
        tangent_out = jnp.einsum('...e,ejk,...kl->...jl', t_dot, gens, u)
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
    """Compute first derivative with respect to (each) matrix input."""
    ts = jax.tree.map(
        lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us)

    ts_basis = _std_basis(ts)
    curve = _local_curve_vec(fun, gens, us)
    jvp = partial(jax.jvp, curve, (ts,))
    out, jac = jax.vmap(jvp, out_axes=(None, -1))((ts_basis,))

    jac_tree = jax.tree.map(
        partial(_jacfwd_unravel, ts), jac)

    return out, jac_tree


def path_grad2(fun, gens, us):
    """Compute first and second derivative with respect to (each) matrix input."""
    curve = _local_curve_vec(fun, gens, us)

    def grad_fn(ts, vec):
        out, tangents = jax.jvp(curve, (ts,), (vec,))
        return tangents, out

    @partial(jax.vmap, in_axes=(None, 0), out_axes=(None, -1, -1))
    def grad2_fn(ts, vec):
        grad, tangents, out = jax.jvp(partial(grad_fn, vec=vec), (ts,), (vec,), has_aux=True)
        return out, grad, tangents

    ts = jax.tree.map(
        lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us)
    ts_basis = _std_basis(ts)

    out, jac, jac2 = grad2_fn(ts, ts_basis)

    jac = jax.tree.map(
        partial(_jacfwd_unravel, ts), jac)

    jac2 = jax.tree.map(
        partial(_jacfwd_unravel, ts), jac2)

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

    ts = jax.tree_util.tree_map(
        lambda u: np.zeros(u.shape[:-2] + (len(gens),)), us
    )
    ts_basis = _std_basis(ts)

    out, tangents = grad2_fn(ts, ts_basis)

    return out.reshape(-1, len(gens)), jnp.sum(tangents).real
