import jax
import jax.numpy as jnp
from einops import einsum
from jax_autovmap import auto_vmap


def roll_lattice(lattice, loc, invert=False):
    """Roll lattice by given tuple.

    Axes are counted from the left, so if the dimension of `lattice` is
    larger than the length of `loc`, the trailing dimensions of the lattice
    are treaded as "channels".

    By default, the lattice is "rolled into position".
    For example, if loc = (1, 1, 0) then the new lattice will have the value
    `rolled_lattice[0, 0, 0, ...] = rolled_lattice[1, 1, 0, ...]`.

    Args:
        lattice: Array with leading exes representing lattice.
        loc: Tuple of integers.
        invert: If true, roll lattice in opposite direction.
    """
    dims = tuple(range(len(loc)))
    if invert:
        lattice = jnp.roll(lattice, loc, dims)
    else:
        lattice = jnp.roll(lattice, tuple(map(lambda i: -i, loc)), dims)
    return lattice


# -- lattice symmetries -- #


def swap_axes(lat, ax0=0, ax1=1):
    # lat shape: (x, ..., y, d, i, j)
    lat = lat.swapaxes(ax0, ax1)
    lat0 = lat[..., ax0, :, :]
    lat1 = lat[..., ax1, :, :]
    lat = lat.at[..., ax0, :, :].set(lat1)
    lat = lat.at[..., ax1, :, :].set(lat0)
    return lat


def flip_axis(lat, axis=0):
    lat = jnp.flip(lat, axis)
    # by convention edge points in "positive" direction
    lat = lat.at[..., axis, :, :].set(
        jnp.roll(lat[..., axis, :, :].conj().swapaxes(-1, -2), -1, axis=axis)
    )
    return lat


def rotate_lat(lat, ax0=0, ax1=1):
    lat = swap_axes(lat, ax0, ax1)
    lat = flip_axis(lat, ax0)
    return lat


def apply_gauge_sym(lat, gs):
    dim = lat.shape[-3]  # lattice dim
    spc = ' '.join(f'l{d}' for d in range(dim))

    for d in range(dim):
        shift = tuple(1 if i == d else 0 for i in range(dim))
        gs_rolled = roll_lattice(gs, shift).conj().swapaxes(-1, -2)
        lat = lat.at[..., d, :, :].set(
            einsum(
                gs, gs_rolled, lat[..., d, :, :],
                f'{spc} i ic, {spc} jc j, ... {spc} ic jc -> ... {spc} i j')
        )

    return lat


# -- Wilson action -- #


def _wilson_log_prob(lat: jax.Array, beta: float) -> jax.Array:

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
    """Computes the log probability for a lattice configuration under the Wilson action.

    This function calculates the sum of the real parts of the traces of all plaquettes
    in the lattice, summed over all planes and lattice sites. The result is
    proportional to the standard Wilson gauge action.

    The lattice `lat` is expected to have a shape compatible with this function's
    internal logic, `(*batch, *space, D, N, N)`, where `*space` represents the
    spatial dimensions, `D` is the lattice dimensionality, and `(N, N)` is the
    shape of the SU(N) matrices.

    For a D-dimensional lattice, plaquettes are computed for each of the
    D*(D-1)/2 planes.

    Args:
        lat: The lattice configuration of SU(N) matrices.
             Shape: `(L_D-1, ..., L_0, D, N, N)`.
        beta: The inverse coupling constant.

    Returns:
        The log probability, a scalar value (same shape as `batch`)
    """
    lat_dim = jnp.shape(lat)[-3]
    return auto_vmap(lat_dim + 3, 0)(_wilson_log_prob)(lat, beta)

def wilson_action(lat: jax.Array, beta: float) -> jax.Array:
    """Computes the Wilson action for a given lattice configuration."""
    return -wilson_log_prob(lat, beta)
