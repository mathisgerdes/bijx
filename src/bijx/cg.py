r"""Crouch-Grossmann integration methods for Lie group ordinary differential equations.

This module implements geometric integration schemes for ODEs on matrix Lie groups,
using the Crouch-Grossmann family of methods. These ensure solutions remain
on the manifold throughout integration (up to numerical accuracy).

Key concepts:
    - Lie group ODEs: Differential equations $\dot{g} = A(t,g) g$ where $g(t) \in G$
    - Crouch-Grossmann schemes: Runge-Kutta-type methods using matrix exponentials
    - Butcher tableaux: Coefficient arrays defining integration schemes

Mathematical background:
For ODEs on matrix Lie groups of the form $\dot{g} = f(t,g)$ where $f$ takes
values in the Lie algebra, Crouch-Grossmann methods approximate:
$g_{n+1} = \exp(\sum_i b_i k_i) g_n$

where $k_i$ are stage vectors computed via the tableau coefficients.
This ensures $g_{n+1} \in G$ whenever $g_n \in G$.
"""

from functools import partial, reduce

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax import core, custom_derivatives


@flax.struct.dataclass
class ButcherTableau:
    r"""Butcher tableau defining coefficients for Runge-Kutta integration schemes.

    Encodes the coefficient structure for explicit Runge-Kutta methods in
    the standard Butcher tableau format. Used to define Crouch-Grossmann
    schemes for integration on Lie groups.

    The tableau has the structure:

    ::

        c_1 | a_11  a_12  ...  a_1s
        c_2 | a_21  a_22  ...  a_2s
         :  |  :     :    ⋱    :
        c_s | a_s1  a_s2  ...  a_ss
        ----+----------------------
            | b_1   b_2   ...  b_s

    For explicit methods: $a_{ij} = 0$ for $j \geq i$.
    Consistency requires: $c_i = \sum_j a_{ij}$ and $\sum_i b_i = 1$.
    """

    stages: int
    """Number of stages $s$ in the method."""

    a: tuple[tuple[int, ...]]
    """Coefficient matrix $(a_{ij})$ as nested tuples."""

    b: tuple[int, ...]
    """Weight vector $(b_i)$ as tuple."""

    c: tuple[int, ...]
    """Node vector $(c_i)$ as tuple (computed from $a$)."""

    @classmethod
    def from_ab(cls, a, b):
        r"""Construct Butcher tableau from coefficient matrix and weights.

        Creates a ButcherTableau instance from the $a$ matrix and $b$ vector,
        automatically computing the node vector $c_i = \sum_j a_{ij}$ and
        validating consistency conditions.

        Args:
            a: Coefficient matrix as list of lists, shape $(s, s)$.
            b: Weight vector as list, length $s$.

        Returns:
            ButcherTableau instance with computed node vector.

        Raises:
            AssertionError: If consistency conditions are violated:
                - Weights don't sum to 1: $\sum_i b_i \neq 1$
                - Method is not explicit: $a_{ij} \neq 0$ for $j \geq i$
                - Dimensions don't match

        Example:
            >>> # Second-order Crouch-Grossmann method
            >>> cg2 = ButcherTableau.from_ab(
            ...     a=[[0, 0], [1/2, 0]], b=[0, 1]
            ... )
        """
        a = tuple(tuple(ai) for ai in a)
        b = tuple(b)
        c = tuple(sum(ai) for ai in a)

        assert all(len(ai) == len(c) for ai in a)
        assert len(b) == len(c)
        assert np.isclose(sum(b), 1)

        for j in range(len(c)):
            for i in range(j + 1):
                assert a[i][j] == 0, "only explicit methods supported"

        return cls(stages=len(c), a=a, b=b, c=c)


EULER = ButcherTableau.from_ab(
    a=[[0]],
    b=[1],
)
r"""Forward Euler method (1st-order, 1 stage).

The simplest integration scheme: $y_{n+1} = y_n + h f(t_n, y_n)$.

For Lie groups: $g_{n+1} = \exp(h A(t_n, g_n)) g_n$.
"""

CG2 = ButcherTableau.from_ab(
    a=[[0, 0], [1 / 2, 0]],
    b=[0, 1],
)
r"""Second-order Crouch-Grossmann method (2nd-order, 2 stages).

A two-stage method achieving second-order accuracy for Lie group ODEs.
This is the Lie group analogue of the classical midpoint rule.

Stages:
1. $k_1 = A(t_n, g_n)$
2. $k_2 = A(t_n + h/2, \exp(h k_1/2) g_n)$

Update: $g_{n+1} = \exp(h k_2) g_n$
"""

CG3 = ButcherTableau.from_ab(
    a=[[0, 0, 0], [3 / 4, 0, 0], [119 / 216, 17 / 108, 0]],
    b=[13 / 51, -2 / 3, 24 / 17],
)
r"""Third-order Crouch-Grossmann method (3rd-order, 3 stages)."""


def transport(vect, z, inverse=False):
    r"""Transport Lie algebra element to tangent space at group element.

    Performs parallel transport of a Lie algebra element (tangent vector
    at identity) to the tangent space at an arbitrary group element.

    For right-invariant vector fields: $X_g = X_e \cdot g$
    For left-invariant vector fields: $X_g = g \cdot X_e$

    Here, we choose the convention of right-invariant vector fields.

    The transport maps vectors from $T_e G$ (Lie algebra) to $T_g G$.

    Args:
        vect: Lie algebra element (tangent vector at identity).
        z: Group element providing the transport destination.
        inverse: If True, transport from $T_g G$ back to $T_e G$.

    Returns:
        Transported tangent vector at the specified group element.

    Example:
        >>> # Transport SU(2) generator to arbitrary group element
        >>> X = bijx.lie.SU2_GEN[0]  # Generator at identity
        >>> g = bijx.lie.sample_haar(rng, n=2)  # Arbitrary SU(2) element
        >>> X_g = transport(X, g)  # Tangent vector at g
        >>> X_g.shape
        (2, 2)
    """
    if inverse:
        z_inv = z.T.conj()
        return jnp.einsum("...ij,...jk->...ik", vect, z_inv)
    return jnp.einsum("...ij,...jk->...ik", vect, z)


def stage_reduce(y0, is_lie, *deltas):
    r"""Accumulate stage increments using appropriate group operation.

    Combines multiple increments using either vector addition (for Euclidean
    spaces) or matrix exponential composition (for Lie groups).

    For Euclidean space: $y = y_0 + \sum_i \delta_i$
    For Lie groups: $g = \left(\prod_i \exp(\delta_i)\right) g_0$

    The Lie group version ensures the result remains on the manifold
    by using the exponential map and group multiplication.

    Args:
        y0: Initial state (group element or vector).
        is_lie: Boolean indicating whether to use Lie group operations.
        *deltas: Sequence of increments to accumulate.

    Returns:
        Updated state after accumulating all increments.
    """
    # Note: Could pass max_squarings argument to expm
    expm = jax.scipy.linalg.expm

    if not is_lie:
        return y0 + sum(deltas)

    return reduce(lambda y, v: jnp.einsum("...ij,...jk->...ik", expm(v), y), deltas, y0)


def cg_stage(y0, vect, is_lie, ai, step_size):
    r"""Compute intermediate state for Crouch-Grossmann stage.

    Combines previous stage vectors according
    to the Butcher tableau coefficients.

    The computation follows:
    $g_i = \left(\prod_j \exp(h a_{ij} k_j)\right) g_0$

    where $k_j$ are stage vectors and $a_{ij}$ are tableau coefficients.

    Args:
        y0: Initial state for the current step.
        vect: List of stage vectors from previous stages.
        is_lie: Boolean tree indicating which components use Lie group operations.
        ai: Row of Butcher tableau coefficients for current stage.
        step_size: Integration step size $h$.

    Returns:
        Intermediate state for evaluating the next stage vector.
    """

    deltas = [
        jax.tree.map(lambda vect_j: step_size * aij * vect_j, vect[j])
        for j, aij in enumerate(ai)
        if aij != 0
    ]

    if len(deltas) == 0:
        return y0
    return jax.tree.map(stage_reduce, y0, is_lie, *deltas)


def crouch_grossmann_step(is_lie, tableau, vector_field, step_size, t, y0):
    r"""Execute single step of Crouch-Grossmann integration method.

    Performs one integration step using the specified Butcher tableau,
    computing all intermediate stages and the final update.

    Args:
        is_lie: Pytree of booleans indicating Lie group vs Euclidean components.
        tableau: ButcherTableau defining the integration method.
        vector_field: Function $(t, g) \mapsto A$ where $A$ is in the Lie algebra.
        step_size: Integration step size $h$.
        t: Current time.
        y0: Current state (group element or vector).

    Returns:
        Updated state after one integration step.

    Important:
        The vector field must return values in the Lie algebra for Lie group
        components, enabling the exponential map to produce valid group elements.
    """
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
    r"""Integrate ODE using Crouch-Grossmann method with custom differentiation.

    Solves: $\dot{g} = f(t, g, \text{args})$ from $t_0$ to $t_1$

    For Lie groups: $\dot{g} = A(t, g) g$ where $A(t, g) \in \mathfrak{g}$
    For Euclidean: $\dot{y} = f(t, y)$ (standard ODE)

    Args:
        vector_field: Function $(t, g, \text{args}) \mapsto A$ where $A$ is
            in the Lie algebra for Lie group components.
        y0: Initial condition (group element or vector).
        args: Additional parameters passed to vector_field.
        t0: Initial time.
        t1: Final time.
        step_size: Integration step size (positive for forward integration).
        is_lie: Boolean tree indicating which components use Lie group operations.
        tableau: ButcherTableau defining the integration method (default: Euler).

    Returns:
        Solution at time $t_1$.

    Example:
        >>> def rigid_body_eqs(t, R, omega):
        ...     # R(t) ∈ SO(3), omega is angular velocity
        ...     Omega = skew_symmetric(omega)
        ...     return Omega  # Lie algebra element
        >>> R0 = jnp.eye(3)  # Initial orientation
        >>> omega = jnp.array([1.0, 0.0, 0.0])  # Rotation about x-axis
        >>> R_final = crouch_grossmann(
        ...     rigid_body_eqs, R0, omega, 0.0, 1.0, 0.01, True, CG2
        ... )

    Important:
        The vector field must return values in the Lie algebra for Lie group
        components, enabling the exponential map to produce valid group elements.
    """
    for arg in jax.tree_util.tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of args must be arrays or scalars, but got {arg}."
            )

    ts = jnp.array([t0, t1], dtype=float)
    converted, consts = custom_derivatives.closure_convert(
        vector_field, ts[0], y0, args
    )
    return _crouch_grossmann(
        is_lie, tableau, converted, step_size, ts, y0, args, *consts
    )


def _bounded_next_time(cur_t, step_size, t_end):
    """Compute next integration time, bounded by endpoint."""
    next_t = cur_t + step_size
    return jnp.where(
        step_size > 0, jnp.minimum(next_t, t_end), jnp.maximum(next_t, t_end)
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _crouch_grossmann(is_lie, tableau, vector_field, step_size, ts, y0, *args):
    r"""Internal Crouch-Grossmann integrator with custom reverse-mode AD.

    The custom VJP implements the adjoint method for ODEs, integrating
    the adjoint equation backwards in time to compute sensitivities.
    """

    def func_(t, y):
        return vector_field(t, y, *args)

    step = partial(crouch_grossmann_step, is_lie, tableau, func_)

    def cond_fun(carry):
        """Check if integration endpoint has been reached."""
        cur_t, cur_y = carry
        return jnp.where(step_size > 0, cur_t < ts[1], cur_t > ts[1])

    def body_fun(carry):
        """Execute one integration step with adaptive step size."""
        cur_t, cur_y = carry
        next_t = _bounded_next_time(cur_t, step_size, ts[1])
        dt = next_t - cur_t
        next_y = step(dt, cur_t, cur_y)
        return next_t, next_y

    init_carry = (ts[0], y0)
    t1, y1 = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    return y1


def _crouch_grossmann_fwd(is_lie, tableau, vector_field, step_size, ts, y0, *args):
    """Forward pass for custom differentiation."""
    y1 = _crouch_grossmann(is_lie, tableau, vector_field, step_size, ts, y0, *args)
    return y1, (ts, y1, args)


def _tree_fill(pytree, val):
    """Fill PyTree structure with constant value."""
    leaves, tree = jax.tree_util.tree_flatten(pytree)
    return tree.unflatten([val] * len(leaves))


def _crouch_grossmann_rev(is_lie, tableau, vector_field, step_size, res, g):
    r"""Reverse-mode differentiation rule for Crouch-Grossmann integration.

    Note:
        The adjoint state is treated as Euclidean for simplicity, though
        more sophisticated cotangent space handling could reduce numerical error.
    """
    ts, y1, args = res

    def _aux(t, y, args):
        vect0 = vector_field(t, y, *args)
        # need to take gradient of actual tangent vector in real space
        # below, so transport vect0 to y for all values of is_lie type.
        vect = jax.tree.map(
            lambda v, y, lie: transport(v, y) if lie else v, vect0, y, is_lie
        )
        return vect, vect0

    def augmented_ode(t, state, args):
        y, adj, *_ = state

        _, vjp, vect = jax.vjp(_aux, t, y, args, has_aux=True)
        t_bar, y_bar, args_bar = jax.tree.map(jnp.negative, vjp(adj))

        return vect, y_bar, t_bar, args_bar

    # effect of moving measurement time
    # need true tangent vectors in embedding space for dot product here
    # (otherwise need more general contraction between vector and cotangent g)
    t_bar = sum(
        map(
            lambda lie, v, vbar, y: jnp.sum((transport(v, y) if lie else v) * vbar),
            jax.tree.leaves(is_lie),
            jax.tree.leaves(vector_field(ts[1], y1, *args)),
            jax.tree.leaves(g),
            jax.tree.leaves(y1),
        )
    )

    t0_bar = -t_bar

    # state = (y, adjoint_state, grad_t, grad_args)
    state = (y1, g, t0_bar, jax.tree.map(jnp.zeros_like, args))

    # NOTE:
    # _tree_fill(is_lie, False) means we treat the adjoint state as simply a
    # Euclidean object in the ambient space; this may be improved in the future
    # to reduce error, since actually it is restricted to the cotangent space.
    aux_is_lie = (is_lie, _tree_fill(is_lie, False), False, _tree_fill(args, False))

    _, y_bar, t0_bar, args_bar = _crouch_grossmann(
        aux_is_lie, tableau, augmented_ode, -step_size, ts[::-1], state, args
    )

    ts_bar = jnp.array([t0_bar, t_bar])
    return (ts_bar, y_bar, *args_bar)


_crouch_grossmann.defvjp(_crouch_grossmann_fwd, _crouch_grossmann_rev)
