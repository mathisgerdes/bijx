"""
Bijections. Not to be exported as a module.
"""

from . import base, continuous, conv_cnf, coupling, fourier, meta, scalar, splines

from .base import *
from .continuous import *
from .conv_cnf import *
from .coupling import *
from .fourier import *
from .meta import *
from .scalar import *
from .splines import *

# Collect __all__ from all submodules to prevent module names from being exported
__all__ = []
for module in [base, continuous, conv_cnf, coupling, fourier, meta, scalar, splines]:
    if hasattr(module, "__all__"):
        __all__.extend(module.__all__)
