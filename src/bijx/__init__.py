r"""Bijections and normalizing flows with JAX, with focus on physics.

This is a library for normalizing flows built on JAX and Flax NNX,
with some specialized tools for lattice field theory. The library
provides flexible bijection primitives, distribution interfaces, and specialized
neural network components for building probabilistic models.

Example:
    >>> # Create a simple normalizing flow
    >>> base_dist = bijx.IndependentNormal(event_shape=(10,), rngs=rngs)
    >>> bijection = bijx.Chain(
    ...     bijx.AffineLinear(rngs=rngs),
    ...     bijx.Tanh(),
    ...     bijx.AffineLinear(rngs=rngs)
    ... )
    >>>
    >>> # Sample and evaluate densities
    >>> x, log_p = base_dist.sample(batch_shape=(100,))
    >>> y, log_q = bijection.forward(x, log_p)
"""

# Submodules that should be imported as submodules
from . import (
    cg,
    fourier,
    lattice,
    lie,
    nn,
)

# Version
try:
    from ._version import __version__
except Exception:
    # Fallback for environments where setuptools_scm didn't populate _version yet
    __version__ = "0.0.dev0"

# Core modules - fully exported to top level
from .bijections import *
from .distributions import *
from .mcmc import *
from .samplers import *
from .solvers import *
from .utils import *
