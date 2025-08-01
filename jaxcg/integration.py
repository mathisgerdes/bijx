from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import core, custom_derivatives

from .tableau import EULER


def transport(vect, z, inverse=False):
    """Transport lie algebra element to tangent space at given point."""
    if inverse:
        z_inv = z.T.conj()
        return jnp.einsum('...ij,...jk->...ik', vect, z_inv)
    return jnp.einsum('...ij,...jk->...ik', vect, z)


def stage_reduce(y0, is_lie, *deltas):
    """Accumulate deltas, sum or expm product."""
    # TODO: could pass max_squarings argument to expm
    expm = jax.scipy.linalg.expm

    if not is_lie:
        return y0 + sum(deltas)

    return reduce(
        lambda y, v: jnp.einsum('...ij,...jk->...ik', expm(v), y),
        deltas, y0)


def cg_stage(y0, vect, is_lie, ai, step_size):
    """Compute intermediate stage state."""

    deltas = [jax.tree_map(lambda l: step_size * aij * l, vect[j])
              for j, aij in enumerate(ai)
              if aij != 0]

    if len(deltas) == 0:
        return y0
    return jax.tree_map(stage_reduce, y0, is_lie, *deltas)


def crouch_grossmann_step(is_lie, tableau, vector_field, step_size, t, y0):
    # all intermediate vectors
    vectors = [None] * tableau.stages

    for i, ai in enumerate(tableau.a):
        # intermediate time for stage i
        ti = t + step_size * tableau.c[i]
        # intermediate state
        intermediate = cg_stage(y0, vectors, is_lie, ai, step_size)
        # evaluate vector field
        vectors[i] = vector_field(ti, intermediate)

    return cg_stage(y0, vectors, is_lie, tableau.b, step_size)


def crouch_grossmann(vector_field, y0, args, t0, t1, step_size, is_lie, tableau=EULER):
    for arg in jax.tree_util.tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f'The contents of args must be arrays or scalars, but got {arg}.')

    ts = jnp.array([t0, t1], dtype=float)
    converted, consts = custom_derivatives.closure_convert(vector_field, ts[0], y0, args)
    return _crouch_grossmann(is_lie, tableau, converted, step_size, ts, y0, args, *consts)


def _bounded_next_time(cur_t, step_size, t_end):
    next_t = cur_t + step_size
    return jnp.where(step_size > 0, jnp.minimum(next_t, t_end), jnp.maximum(next_t, t_end))


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _crouch_grossmann(is_lie, tableau, vector_field, step_size, ts, y0, *args):
    func_ = lambda t, y: vector_field(t, y, *args)

    step = partial(crouch_grossmann_step, is_lie, tableau, func_)

    def cond_fun(carry):
        """Check if we've reached the last time point."""
        cur_t, cur_y = carry
        return jnp.where(step_size > 0, cur_t < ts[1], cur_t > ts[1])

    def body_fun(carry):
        """Take one step of RK4."""
        cur_t, cur_y = carry
        next_t = _bounded_next_time(cur_t, step_size, ts[1])
        dt = next_t - cur_t
        next_y = step(dt, cur_t, cur_y)
        return next_t, next_y

    init_carry = (ts[0], y0)
    t1, y1 = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return y1


def _crouch_grossmann_fwd(is_lie, tableau, vector_field, step_size, ts, y0, *args):
    y1 = _crouch_grossmann(is_lie, tableau, vector_field, step_size, ts, y0, *args)
    return y1, (ts, y1, args)


def _tree_fill(pytree, val):
    leaves, tree = jax.tree_util.tree_flatten(pytree)
    return tree.unflatten([val] * len(leaves))


def _crouch_grossmann_rev(is_lie, tableau, vector_field, step_size, res, g):
    ts, y1, args = res

    def _aux(t, y, args):
        vect0 = vector_field(t, y, *args)
        # need to take gradient of actual tangent vector in real space
        # below, so transport vect0 to y for all values of is_lie type.
        vect = jax.tree_map(
            lambda v, y, lie: transport(v, y) if lie else v,
            vect0, y, is_lie)
        return vect, vect0

    def augmented_ode(t, state, args):
        y, adj, *_ = state

        _, vjp, vect = jax.vjp(_aux, t, y, args, has_aux=True)
        t_bar, y_bar, args_bar = jax.tree_map(jnp.negative, vjp(adj))

        return vect, y_bar, t_bar, args_bar

    # effect of moving measurement time
    # need true tangent vectors in embedding space for dot product here
    # (otherwise need more general contraction between vector and cotangent g)
    t_bar = sum(map(lambda l, v, vbar, y: jnp.sum((transport(v, y) if l else v) * vbar),
                    jax.tree_util.tree_leaves(is_lie),
                    jax.tree_util.tree_leaves(vector_field(ts[1], y1, *args)),
                    jax.tree_util.tree_leaves(g),
                    jax.tree_util.tree_leaves(y1)))

    t0_bar = -t_bar

    # state = (y, adjoint_state, grad_t, grad_args)
    state = (y1, g, t0_bar, jax.tree_map(jnp.zeros_like, args))

    # TODO
    # _tree_fill(is_lie, False) means we treat the adjoint state as simply a
    # Euclidean object in the ambient space; this may be improved in the future
    # to reduce error, since actually it lies in the cotangent space.
    aux_is_lie = (is_lie, _tree_fill(is_lie, False), False, _tree_fill(args, False))

    _, y_bar, t0_bar, args_bar = _crouch_grossmann(
        aux_is_lie, tableau, augmented_ode, -step_size, ts[::-1], state, args)

    ts_bar = jnp.array([t0_bar, t_bar])
    return (ts_bar, y_bar, *args_bar)


_crouch_grossmann.defvjp(_crouch_grossmann_fwd, _crouch_grossmann_rev)
