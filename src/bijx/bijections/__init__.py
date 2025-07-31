"""
Bijections package.
"""

from .base import Bijection, ApplyBijection, Chain, ScanChain, Frozen, Inverse
from .continuous import ContFlowDiffrax, ContFlowRK4
from .coupling import (
    checker_mask,
    ModuleReconstructor,
    BinaryMask,
    GeneralCouplingLayer,
    AutoVmapReconstructor,
)
from .fourier import SpectrumScaling, FreeTheoryScaling, ToFourierData
from .meta import MetaLayer, ExpandDims, SqueezeDims, Reshape
from .scalar import *
from .splines import (
    MonotoneRQSpline,
    rational_quadratic_spline,
)

__all__ = [
    # Base classes
    "Bijection",
    "ApplyBijection",
    "Chain",
    "ScanChain",
    "Frozen",
    "Inverse",
    # Continuous flows
    "ContFlowDiffrax",
    "ContFlowRK4",
    # Coupling layers
    "checker_mask",
    "ModuleReconstructor",
    "BinaryMask",
    "GeneralCouplingLayer",
    "AutoVmapReconstructor",
    # Fourier-space bijections
    "SpectrumScaling",
    "FreeTheoryScaling",
    "ToFourierData",
    # Linear transformations
    "Scaling",
    "ScalarBijection",
    "Shift",
    "AffineLinear",
    # Meta transformations
    "MetaLayer",
    "ExpandDims",
    "SqueezeDims",
    "Reshape",
    # Splines
    "MonotoneRQSpline",
    "rational_quadratic_spline",
    # Fourier
    "Fourier",
    "FourierBasis",
    "FourierSeries",
]
