"""
Methods for manipulating lattice field configurations.

Includes both scalar and gauge fields, generally assuming periodic boundary conditions.
"""

from . import scalar, gauge

# only export things where application domain (e.g. gauge vs scalar) is clear
# or is universally applicable.
from .gauge import (
    apply_gauge_sym,
    wilson_log_prob,
    wilson_action,
    roll_lattice,
)


__all__ = [
    # submodules
    "scalar",
    "gauge",
    # functions
    "apply_gauge_sym",
    "wilson_log_prob",
    "wilson_action",
    "roll_lattice",
]
