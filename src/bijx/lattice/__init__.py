"""
Methods for manipulating lattice field configurations.

Includes both scalar and gauge fields, generally assuming periodic boundary conditions.
"""

from . import scalar, gauge

from .general import (
    roll_lattice,
)


__all__ = [
    # submodules
    "scalar",
    "gauge",
    # general
    "roll_lattice",
]
