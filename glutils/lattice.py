import jax.numpy as jnp
from einops import einsum


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