"""Gauge field theory utilities.

This module provides computational tools for SU(N) gauge theories on discrete
lattices with periodic boundary conditions.
"""

import jax
import jax.numpy as jnp
from einops import einsum
from jax_autovmap import autovmap

from .general import roll_lattice

# -- Lattice Symmetries -- #


def swap_axes(lat, ax0=0, ax1=1):
    r"""Apply lattice coordinate transformation swapping two spatial axes.

    Performs a coordinate swap transformation on gauge field configurations,
    properly handling both the spatial coordinate permutation and the
    corresponding gauge field index transformation.

    Args:
        lat: Gauge field configuration of shape $(L_0, ..., L_{d-1}, d, N, N)$
            where the last three dimensions are (direction, matrix_row, matrix_col).
        ax0: First spatial axis to swap (default: 0).
        ax1: Second spatial axis to swap (default: 1).

    Returns:
        Transformed gauge field configuration with swapped coordinates and
        corresponding gauge field indices.

    Note:
        This is part of the discrete symmetry group of the lattice, preserving
        the gauge action under coordinate transformations.
    """
    # lat shape: (x, ..., y, d, i, j)
    lat = lat.swapaxes(ax0, ax1)
    lat0 = lat[..., ax0, :, :]
    lat1 = lat[..., ax1, :, :]
    lat = lat.at[..., ax0, :, :].set(lat1)
    lat = lat.at[..., ax1, :, :].set(lat0)
    return lat


def flip_axis(lat, axis: int):
    r"""Apply lattice reflection transformation along specified axis.

    Performs a parity transformation (reflection) along one spatial axis,
    properly handling the gauge field transformation under coordinate inversion.
    Under reflection :math:`x_\mu \to -x_\mu`, gauge links transform as:
    :math:`U_\mu(x) \to U_\mu^\dagger(x - \hat{\mu})`

    This ensures gauge invariance is preserved under the reflection symmetry.

    Args:
        lat: Gauge field configuration of shape $(L_0, ..., L_{d-1}, d, N, N)$.
        axis: Spatial axis along which to apply the reflection.

    Returns:
        Reflected gauge field configuration with proper gauge field
        transformation applied.

    Example:
        >>> # Apply reflection symmetry to 2D gauge field
        >>> lat_reflected = flip_axis(lat, axis=0)
        >>> # Wilson action should be preserved
        >>> S_orig = wilson_action(lat, beta=1.0)
        >>> S_refl = wilson_action(lat_reflected, beta=1.0)
        >>> jnp.allclose(S_orig, S_refl)
        Array(True, dtype=bool)
    """
    lat = jnp.flip(lat, axis)
    # by convention edge points in "positive" direction
    lat = lat.at[..., axis, :, :].set(
        jnp.roll(lat[..., axis, :, :].conj().swapaxes(-1, -2), -1, axis=axis)
    )
    return lat


def rotate_lat(lat, ax0=0, ax1=1):
    r"""Apply 90-degree rotation in the plane spanned by two spatial axes.

    Performs a discrete rotation transformation by 90 degrees in the plane
    defined by axes $(ax0, ax1)$. This combines a coordinate swap followed
    by a reflection to achieve the rotation.

    The transformation maps $(x_{ax0}, x_{ax1}) \to (-x_{ax1}, x_{ax0})$,
    implementing a counterclockwise rotation by 90 degrees.

    Args:
        lat: Gauge field configuration of shape $(L_0, ..., L_{d-1}, d, N, N)$.
        ax0: First axis defining the rotation plane.
        ax1: Second axis defining the rotation plane.

    Returns:
        Rotated gauge field configuration with proper gauge field
        transformations applied.

    Example:
        >>> # Four 90-degree rotations should return to original
        >>> lat_orig = lat
        >>> for _ in range(4):
        ...     lat = rotate_lat(lat, 0, 1)
        >>> jnp.allclose(lat, lat_orig)
        Array(True, dtype=bool)
    """
    lat = swap_axes(lat, ax0, ax1)
    lat = flip_axis(lat, ax0)
    return lat


def apply_gauge_sym(lat, gs):
    r"""Apply gauge transformation to lattice gauge field configuration.

    Performs a local SU(N) gauge transformation on the gauge field configuration.
    Under a gauge transformation :math:`g(x) \in SU(N)`, gauge links transform as:
    :math:`U_\mu(x) \to g(x) U_\mu(x) g^\dagger(x + \hat{\mu})`

    Args:
        lat: Gauge field configuration of shape $(L_0, ..., L_{d-1}, d, N, N)$.
        gs: Gauge transformation matrices of shape $(L_0, ..., L_{d-1}, N, N)$.

    Returns:
        Gauge-transformed field configuration with the same shape as input.

    Example:
        >>> # Gauge transformation preserves Wilson action
        >>> S_orig = wilson_action(lat, beta=1.0)
        >>> lat_transformed = apply_gauge_sym(lat, gauge_matrices)
        >>> S_transformed = wilson_action(lat_transformed, beta=1.0)
        >>> jnp.allclose(S_orig, S_transformed)
        Array(True, dtype=bool)
    """
    dim = lat.shape[-3]  # lattice dim
    spc = " ".join(f"l{d}" for d in range(dim))

    for d in range(dim):
        shift = tuple(1 if i == d else 0 for i in range(dim))
        gs_rolled = roll_lattice(gs, shift).conj().swapaxes(-1, -2)
        lat = lat.at[..., d, :, :].set(
            einsum(
                gs,
                gs_rolled,
                lat[..., d, :, :],
                f"{spc} i ic, {spc} jc j, ... {spc} ic jc -> ... {spc} i j",
            )
        )

    return lat


# -- Wilson Action -- #


def _wilson_log_prob(lat: jax.Array, beta: float) -> jax.Array:
    r"""Fundamental implementation of Wilson action log probability.

    Compute the log probability under the Wilson gauge action for a single
    lattice configuration. This is the core computational kernel that
    evaluates all plaquette contributions to the action.

    The Wilson action is:
    $S_W = -\frac{\beta}{N} \sum_{\text{plaq}} \text{Re} \text{Tr}(U_{\mu\nu}(x))$

    where the sum runs over all elementary plaquettes $U_{\mu\nu}(x)$.
    """

    n_mat = lat.shape[-1]
    dim = lat.shape[-3]

    if jnp.ndim(lat) != dim + 3:
        raise ValueError(
            f"Unexpected lattice shape {lat.shape}. "
            f"Expected {dim + 3} dimensions inside vmap: "
            f"({dim} spatial dims, D, N, N)."
        )

    total_action_density = 0.0

    # Iterate over all planes (mu, nu) with mu < nu
    for mu in range(dim):
        for nu in range(mu + 1, dim):
            # U_mu(n)
            u_mu_n = lat[..., mu, :, :]

            # U_nu(n)
            u_nu_n = lat[..., nu, :, :]

            # U_nu(n+e_mu)
            shift_mu = tuple(1 if i == mu else 0 for i in range(dim))
            u_nu_n_plus_emu = roll_lattice(u_nu_n, shift_mu)

            # U_mu(n+e_nu)
            shift_nu = tuple(1 if i == nu else 0 for i in range(dim))
            u_mu_n_plus_enu = roll_lattice(u_mu_n, shift_nu)

            # Plaquette P = U_mu(n) U_nu(n+e_mu) U_mu(n+e_nu)^dagger U_nu(n)^dagger
            plaquette_trace = einsum(
                u_mu_n,
                u_nu_n_plus_emu,
                u_mu_n_plus_enu.conj().swapaxes(-1, -2),
                u_nu_n.conj().swapaxes(-1, -2),
                "... i j, ... j k, ... k l, ... l i -> ...",
            )

            total_action_density += jnp.sum(plaquette_trace.real)

    return (beta / n_mat) * total_action_density


def wilson_log_prob(lat: jax.Array, beta: float) -> jax.Array:
    r"""Compute log probability under Wilson gauge action.

    Evaluates the Wilson gauge action for SU(N) lattice gauge theory:

    $$\log P[U] = \frac{\beta}{N} \sum_{x,\mu<\nu} \text{Re} \text{Tr}[U_{\mu\nu}(x)]$$

    where $U_{\mu\nu}(x)$ is the elementary plaquette:

    .. math::
        U_{\mu\nu}(x) =
        U_\mu(x) U_\nu(x+\hat{\mu}) U_\mu^\dagger(x+\hat{\nu}) U_\nu^\dagger(x)

    Args:
        lat: Gauge field configuration with shape $(L_0, ..., L_{d-1}, d, N, N)$.
        beta: Inverse coupling constant $\beta=2N/g^2$ where $g$ is the gauge coupling.

    Returns:
        Log probability for each configuration in the batch, with shape
        matching the leading (batch) dimensions of the input.

    Example:
        >>> # Single 4x4 SU(2) configuration
        >>> lat = jax.random.normal(key, (4, 4, 2, 2, 2))
        >>> # Ensure SU(2) matrices (simplified for example)
        >>> log_prob = wilson_log_prob(lat, beta=2.4)
        >>> log_prob.shape
        ()
    """
    lat_dim = jnp.shape(lat)[-3]
    return autovmap(lat_dim + 3, 0)(_wilson_log_prob)(lat, beta)


def wilson_action(lat: jax.Array, beta: float) -> jax.Array:
    r"""Compute Wilson gauge action for lattice gauge theory.

    Evaluates the Wilson action $S_W[U] = -\log P[U]$ where $P[U]$
    is the probability weight from :func:`wilson_log_prob`.

    Args:
        lat: Gauge field configuration (see :func:`wilson_log_prob`).
        beta: Inverse coupling constant.

    Returns:
        Wilson action values with shape matching the batch dimensions.

    Note:
        This is simply the negative of :func:`wilson_log_prob`, following
        the convention $\text{action} = -\log(\text{probability})$.

    Example:
        >>> action = wilson_action(lat, beta=2.4)
        >>> # Action increases with disorder (larger beta favors order)
        >>> log_prob = wilson_log_prob(lat, beta=2.4)
        >>> jnp.allclose(action, -log_prob)
        Array(True, dtype=bool)
    """
    return -wilson_log_prob(lat, beta)
