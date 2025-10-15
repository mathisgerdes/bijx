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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import core, custom_derivatives
from jax_autovmap import autovmap


class ButcherTableau(nnx.Pytree):
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

    def __init__(self, a, b):
        r"""Construct Butcher tableau from coefficient matrix and weights.

        Creates a ButcherTableau instance from the $a$ matrix and $b$ vector,
        automatically computing the node vector $c_i = \sum_j a_{ij}$ and
        validating consistency conditions.

        Args:
            a: Coefficient matrix or list of lists, shape $(s, s)$.
            b: Weight vector or list, length $s$.

        Returns:
            ButcherTableau instance with computed node vector.

        Raises:
            AssertionError: If consistency conditions are violated:
                - Weights don't sum to 1: $\sum_i b_i \neq 1$
                - Method is not explicit: $a_{ij} \neq 0$ for $j \geq i$
                - Dimensions don't match

        Example:
            >>> # Second-order Crouch-Grossmann method
            >>> cg2 = ButcherTableau(
            ...     a=[[0, 0], [1/2, 0]], b=[0, 1]
            ... )
        """
        a = tuple(tuple(float(aij) for aij in ai) for ai in a)
        b = tuple(float(bi) for bi in b)
        c = tuple(sum(ai) for ai in a)

        assert all(len(ai) == len(c) for ai in a)
        assert len(b) == len(c)
        assert np.isclose(sum(b), 1)

        for j in range(len(c)):
            for i in range(j + 1):
                assert a[i][j] == 0, "only explicit methods supported"

        self.stages = len(c)
        self.a = nnx.static(a)
        self.b = nnx.static(b)
        self.c = nnx.static(c)


EULER = ButcherTableau(
    a=[[0]],
    b=[1],
)
r"""Forward Euler method (1st-order, 1 stage).

The simplest integration scheme: $y_{n+1} = y_n + h f(t_n, y_n)$.

For Lie groups: $g_{n+1} = \exp(h A(t_n, g_n)) g_n$.
"""

CG2 = ButcherTableau(
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

CG3 = ButcherTableau(
    a=[[0, 0, 0], [3 / 4, 0, 0], [119 / 216, 17 / 108, 0]],
    b=[13 / 51, -2 / 3, 24 / 17],
)
r"""Third-order Crouch-Grossmann method (3rd-order, 3 stages)."""


class ManifoldType(nnx.Pytree):
    r"""Base class defining manifold structure and operations for ODE integration.

    This abstract class specifies how to perform integration steps on different
    types of manifolds (Euclidean spaces, Lie groups, etc.). Subclasses define
    the specific operations needed for their manifold structure.

    Key operations:
        - reduce: Accumulate increments using manifold-appropriate operations
        - post_stage/post_step: Optional projection back to manifold after updates
        - Adjoint methods: Handle cotangent space operations for differentiation

    Common subclasses:
        - Euclidean: Standard vector spaces with addition
        - Matrix: Matrix Lie groups using exponential map
        - Unitary: Unitary groups with optional projection
    """

    def derivative_type(self):
        """Return the ManifoldType for tangent/cotangent spaces.

        For most manifolds, derivatives live in a different space than the
        base manifold. This method specifies the manifold type for these spaces.
        """
        raise NotImplementedError()

    def reduce(self, x, *deltas):
        """Accumulate increments on the manifold.

        Args:
            x: Base point on manifold.
            *deltas: Sequence of increments to accumulate.

        Returns:
            Updated point after accumulating all increments.
        """
        return sum(deltas, start=x)

    def post_stage(self, x):
        """Project state back to manifold after each internal stage.

        Args:
            x: State after stage update.

        Returns:
            Projected state (identity by default).
        """
        return x

    def post_step(self, x):
        """Project state back to manifold after each full integration step.

        Args:
            x: State after full step.

        Returns:
            Projected state (identity by default).
        """
        return x

    def post_adjoint_vjp(self, x, adj_vec, v, adj):
        """Adjust adjoint gradient after VJP computation.

        Args:
            x: Current state.
            adj_vec: Adjoint vector from VJP.
            v: Vector field value.
            adj: Current adjoint state.

        Returns:
            Adjusted adjoint vector.
        """
        return adj_vec

    def pre_adjoint_vjp(self, x, v):
        """Transform vector field before adjoint VJP computation.

        Args:
            x: Current state.
            v: Vector field value at identity/origin.

        Returns:
            Transformed vector (identity by default).
        """
        return v

    def project_adjoint(self, x, adj):
        """Project adjoint state to cotangent space.

        Args:
            x: Current state.
            adj: Adjoint state.

        Returns:
            Projected adjoint (identity by default).
        """
        return adj


class Euclidean(ManifoldType):
    """Euclidean space with standard vector addition.

    Use for standard ODEs in flat space where derivatives are tangent vectors
    and integration uses standard addition. This is the default for most
    non-geometric problems.
    """

    def derivative_type(self):
        return Euclidean()


def _matrix_reduce(x, v):
    """Reduce matrix Lie group element by exponential map and multiplication.

    Computes exp(v) @ x for matrix Lie groups, where v is a Lie algebra element.

    Args:
        x: Matrix group element.
        v: Lie algebra element (tangent vector at identity).

    Returns:
        exp(v) @ x, the group element after applying the exponential update.
    """
    # Note: Could pass max_squarings argument to expm
    expm = jax.scipy.linalg.expm
    return jnp.einsum("...ij,...jk->...ik", expm(v), x)


class Matrix(ManifoldType):
    """Matrix Lie groups using exponential map integration.

    Handles integration on matrix Lie groups (SO(N), SU(N), etc.) using the
    exponential map to ensure solutions remain on the manifold.

    Args:
        right_invariant: Use right-invariant vector fields (X_g = X_e @ g).
            Left-invariant not yet implemented.
        transport_adjoint: If True, keep adjoint in dual tangent space at
            identity. If False, adjoint lives in ambient cotangent space.
    """

    def __init__(self, right_invariant=True, transport_adjoint=False):
        self.right_invariant = right_invariant
        if not right_invariant:
            raise NotImplementedError()

        # if true, keep adjoint in dual tangent space at identity
        self.transport_adjoint = transport_adjoint

    def derivative_type(self):
        return MatrixDeriv(right_invariant=self.right_invariant)

    def reduce(self, x, *deltas):
        return reduce(_matrix_reduce, deltas, x)

    def transport(self, x, v):
        r"""Transport Lie algebra element to tangent space at group element.

        Performs parallel transport of a Lie algebra element (tangent vector
        at identity) to the tangent space at an arbitrary group element.

        For right-invariant vector fields: $X_g = X_e \cdot g$
        For left-invariant vector fields: $X_g = g \cdot X_e$

        This implementation uses the right-invariant convention.
        The transport maps vectors from $T_e G$ (Lie algebra) to $T_g G$.

        Args:
            x: Group element.
            v: Lie algebra element (vector at identity).

        Returns:
            Transported vector at x.
        """
        if self.right_invariant:
            return jnp.einsum("...ij,...jk->...ik", v, x)
        else:
            return jnp.einsum("...ij,...jk->...ik", x, v)

    def pre_adjoint_vjp(self, x, v):
        # if not transporting adjoint, need to transport v to "true" tangent space
        if self.transport_adjoint:
            return v
        return self.transport(x, v)

    def project_adjoint(self, x, adj):
        if self.transport_adjoint:
            return jnp.einsum("...ij,...kj->...ik", adj, x)
        return adj

    def post_adjoint_vjp(self, x, adj_vec, v, adj):
        # if transporting adjoint, add term to keep at origin
        if self.transport_adjoint:
            out = jnp.einsum("...ij,...kj->...ik", adj_vec, x)
            out += jnp.einsum("...ij,...kj->...ik", adj, v)
            out -= jnp.einsum("...ji,...jk->...ik", v, adj)
            return out
        return adj_vec


class MatrixDeriv(ManifoldType):
    """Tangent and cotangent spaces for matrix Lie groups.

    Used for both tangent and cotangent spaces since the projections
    for groups like SU(N) are the same in both cases.

    Args:
        right_invariant: Matches parent Matrix convention.
    """

    def __init__(self, right_invariant=True):
        self.right_invariant = right_invariant

    def derivative_type(self):
        return self


class UnitaryDeriv(ManifoldType):
    """Tangent/cotangent spaces for unitary groups with Lie algebra projection.

    Projects vectors to the Lie algebra (anti-Hermitian matrices for U(N),
    traceless anti-Hermitian for SU(N)) at each stage or step.

    Args:
        project_stage: Project to Lie algebra after each internal stage.
        project_step: Project to Lie algebra after each full step.
        special: If True, enforce traceless condition (for SU(N)).
    """

    def __init__(
        self,
        project_stage=False,
        project_step=False,
        special=True,
    ):
        self.project_stage = project_stage
        # avoid redundant projection
        self.project_step = project_step and not project_stage
        self.special = special

    def derivative_type(self):
        return self

    @autovmap(x=2)
    def project(self, x):
        if self.special:
            trace = jnp.linalg.trace(x)
            x = x - trace / x.shape[-1]
        x = (x - x.mT.conj()) / 2
        return x

    def post_stage(self, x):
        if self.project_stage:
            return self.project(x)
        return x

    def post_step(self, x):
        if self.project_step:
            return self.project(x)
        return x


class Unitary(Matrix):
    """Unitary groups U(N) or SU(N) with optional projection to manifold.

    Extends Matrix with projection operations that enforce unitarity constraints
    using polar decomposition after integration steps.

    Args:
        right_invariant: Use right-invariant vector fields.
        transport_adjoint: Keep adjoint in tangent space at identity.
        project_step: Project back to unitary manifold after each full step.
        project_stage: Project back to unitary manifold after each internal stage.
        special: If True, work with SU(N) (special unitary, det=1).
        derivative: ManifoldType for tangent/cotangent spaces (default: UnitaryDeriv).
    """

    def __init__(
        self,
        right_invariant=True,
        transport_adjoint=False,
        project_step=False,
        project_stage=False,
        special=True,
        derivative=UnitaryDeriv(),
    ):
        super().__init__(right_invariant, transport_adjoint)
        self.project_stage = project_stage
        self.project_step = project_step and not project_stage
        self.special = special
        self._derivative = derivative

        if getattr(derivative, "project_step", False) or getattr(
            derivative, "project_stage", False
        ):
            assert (
                self.transport_adjoint
            ), "adjoint projection not implemented for ambient space adjoint"

    def derivative_type(self):
        return self._derivative

    @autovmap(m=2)
    def project(self, m):
        # Polar decomposition via SVD: m = U S V^H
        u, _, vh = jnp.linalg.svd(m)
        # Construct unitary part of polar decomposition
        u = u @ vh

        if self.special:
            # Normalize to make determinant exactly 1
            det_u = jnp.linalg.det(u)
            det_root = det_u ** (1 / m.shape[-1])
            u = u / det_root

        return u

    def post_stage(self, x):
        if self.project_stage:
            return self.project(x)
        return x

    def post_step(self, x):
        if self.project_step:
            return self.project(x)
        return x


def cg_stage(y0, vect, manifold_types, ai, step_size):
    r"""Compute intermediate state for Crouch-Grossmann stage.

    Combines previous stage vectors according
    to the Butcher tableau coefficients.

    The computation follows:
    $g_i = \left(\prod_j \exp(h a_{ij} k_j)\right) g_0$

    where $k_j$ are stage vectors and $a_{ij}$ are tableau coefficients.

    Args:
        y0: Initial state for the current step.
        vect: List of stage vectors from previous stages.
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
    return jax.tree.map(
        lambda state, mtype, *deltas: mtype.reduce(state, *deltas),
        y0,
        manifold_types,
        *deltas,
    )


def crouch_grossmann_step(manifold_types, tableau, vector_field, step_size, t, y0):
    r"""Execute single step of Crouch-Grossmann integration method.

    Performs one integration step using the specified Butcher tableau,
    computing all intermediate stages and the final update.

    Args:
        manifold_types: Pytree of ManifoldType instances defining manifold structure.
        tableau: ButcherTableau defining the integration method.
        vector_field: Function $(t, g) \mapsto A$ where $A$ is in the Lie algebra
            (for Lie groups) or tangent vector (for Euclidean).
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
        intermediate = cg_stage(y0, vectors, manifold_types, ai, step_size)
        # evaluate vector field
        vectors[i] = vector_field(ti, intermediate)

    return cg_stage(y0, vectors, manifold_types, tableau.b, step_size)


def crouch_grossmann(
    vector_field,
    y0,
    args,
    t0,
    t1,
    step_size,
    manifold_types=None,
    args_types=None,
    tableau=EULER,
):
    r"""Integrate ODE using Crouch-Grossmann method with custom differentiation.

    Solves: $\dot{g} = f(t, g, \text{args})$ from $t_0$ to $t_1$

    For Lie groups: $\dot{g} = A(t, g) g$ where $A(t, g) \in \mathfrak{g}$
    For Euclidean: $\dot{y} = f(t, y)$ (standard ODE)

    Args:
        vector_field: Function $(t, g, \text{args}) \mapsto A$ where $A$ is
            in the Lie algebra for Lie group components, or tangent vector
            for Euclidean components.
        y0: Initial condition (group element or vector).
        args: Additional parameters passed to vector_field.
        t0: Initial time.
        t1: Final time.
        step_size: Integration step size (positive for forward integration).
        manifold_types: ManifoldType or pytree of ManifoldTypes specifying the
            manifold structure of y0. Defaults to Unitary().
        args_types: ManifoldType or pytree of ManifoldTypes for args.
            Defaults to Euclidean().
        tableau: ButcherTableau defining the integration method (default: EULER).

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
        ...     rigid_body_eqs, R0, omega, 0.0, 1.0, 0.01,
        ...     manifold_types=Matrix(), args_types=Euclidean(), tableau=CG2
        ... )

    Important:
        The vector field must return values in the Lie algebra for Lie group
        components, enabling the exponential map to produce valid group elements.
    """
    manifold_types = Unitary() if manifold_types is None else manifold_types
    args_types = Euclidean() if args_types is None else args_types

    for arg in jax.tree_util.tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of args must be arrays or scalars, but got {arg}."
            )

    # make sure both types are matching pytrees
    if isinstance(manifold_types, ManifoldType):
        manifold_types = jax.tree.map(lambda _: manifold_types, y0)
    if isinstance(args_types, ManifoldType):
        args_types = jax.tree.map(lambda _: args_types, args)

    ts = jnp.array([t0, t1], dtype=float)
    converted, consts = custom_derivatives.closure_convert(
        vector_field, ts[0], y0, args
    )

    consts_types = jax.tree.map(lambda _: Euclidean(), consts)
    args_types = (args_types, *consts_types)

    return _crouch_grossmann(
        manifold_types, args_types, tableau, converted, step_size, ts, y0, args, *consts
    )


def _bounded_next_time(cur_t, step_size, t_end):
    """Compute next integration time, bounded by endpoint."""
    next_t = cur_t + step_size
    return jnp.where(
        step_size > 0, jnp.minimum(next_t, t_end), jnp.maximum(next_t, t_end)
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _crouch_grossmann(
    manifold_types, args_types, tableau, vector_field, step_size, ts, y0, *args
):
    r"""Internal Crouch-Grossmann integrator with custom reverse-mode AD.

    The custom VJP implements the adjoint method for ODEs, integrating
    the adjoint equation backwards in time to compute sensitivities.
    """

    def func_(t, y):
        return vector_field(t, y, *args)

    step = partial(crouch_grossmann_step, manifold_types, tableau, func_)

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


def _crouch_grossmann_fwd(
    manifold_types, args_types, tableau, vector_field, step_size, ts, y0, *args
):
    """Forward pass for custom differentiation."""
    y1 = _crouch_grossmann(
        manifold_types, args_types, tableau, vector_field, step_size, ts, y0, *args
    )
    return y1, (ts, y1, args)


def _crouch_grossmann_rev(
    manifold_types, args_types, tableau, vector_field, step_size, res, g
):
    r"""Reverse-mode differentiation rule for Crouch-Grossmann integration.

    Note:
        The adjoint state is treated as Euclidean for simplicity, though
        more sophisticated cotangent space handling could reduce numerical error.
    """
    ts, y1, args = res

    g = jax.tree.map(
        lambda y, adj, mtype: mtype.project_adjoint(y, adj),
        y1,
        g,
        manifold_types,
    )

    def _aux(t, y, args):
        vect0 = vector_field(t, y, *args)
        vect = jax.tree.map(
            lambda v, y, mtype: mtype.pre_adjoint_vjp(y, v), vect0, y, manifold_types
        )
        return vect, vect0

    def augmented_ode(t, state, args):
        y, adj, *_ = state

        _, vjp, vect = jax.vjp(_aux, t, y, args, has_aux=True)
        t_bar, y_bar, args_bar = jax.tree.map(jnp.negative, vjp(adj))

        y_bar = jax.tree.map(
            lambda y_bar, y, v, adj, mtype: mtype.post_adjoint_vjp(y, y_bar, v, adj),
            y_bar,
            y,
            vect,
            adj,
            manifold_types,
        )

        return vect, y_bar, t_bar, args_bar

    # effect of moving measurement time
    t_bar = sum(
        map(
            lambda v, adj, y, mtype: jnp.sum(mtype.pre_adjoint_vjp(y, v) * adj),
            jax.tree.leaves(vector_field(ts[1], y1, *args)),
            jax.tree.leaves(g),
            jax.tree.leaves(y1),
            jax.tree.structure(y1).flatten_up_to(manifold_types),
        )
    )

    t0_bar = -t_bar

    # state = (y, adjoint_state, grad_t, grad_args)
    state = (y1, g, t0_bar, jax.tree.map(jnp.zeros_like, args))

    state_types = (
        manifold_types,
        jax.tree.map(lambda _, mtype: mtype.derivative_type(), y1, manifold_types),
        Euclidean(),
        jax.tree.map(lambda _, mtype: mtype.derivative_type(), args, args_types),
    )

    _, y_bar, t0_bar, args_bar = _crouch_grossmann(
        state_types,
        args_types,
        tableau,
        augmented_ode,
        -step_size,
        ts[::-1],
        state,
        args,
    )

    ts_bar = jnp.array([t0_bar, t_bar])
    return (ts_bar, y_bar, *args_bar)


_crouch_grossmann.defvjp(_crouch_grossmann_fwd, _crouch_grossmann_rev)
