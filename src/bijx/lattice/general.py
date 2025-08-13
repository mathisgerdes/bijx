"""General lattice field theory utilities and operations.

This module provides basic lattice manipulation functions that are common
across different field theories, such as shifting fields along lattice directions.
"""

import jax.numpy as jnp


def roll_lattice(lattice, loc, invert=False):
    r"""Roll lattice fields by specified displacement with periodic boundaries.

    Performs a cyclic shift of lattice data, treating leading dimensions as
    spatial coordinates and trailing dimensions as field components ("channels").

    The default behavior "rolls into position": to access the field value
    originally at site $(x+\text{loc})$, access the rolled lattice at site $x$.

    Args:
        lattice: Lattice field array where leading dimensions correspond to
            spatial coordinates and trailing dimensions are field components.
        loc: Displacement tuple specifying the shift in each spatial direction.
            Length must not exceed the number of spatial dimensions.
        invert: If True, roll in the opposite direction (equivalent to
            accessing the field at $x-\text{loc}$ when looking at site $x$).

    Returns:
        Shifted lattice array with the same shape as input.

    Example:
        >>> lattice = jnp.arange(16).reshape(4, 4)
        >>> shifted = roll_lattice(lattice, (0, 1))
        >>> shifted[0, 0] == lattice[0, 1]
        Array(True, dtype=bool)
    """
    dims = tuple(range(len(loc)))
    if invert:
        lattice = jnp.roll(lattice, loc, dims)
    else:
        lattice = jnp.roll(lattice, tuple(map(lambda i: -i, loc)), dims)
    return lattice
