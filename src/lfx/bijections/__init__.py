"""
Bijections package.
"""

from .base import Bijection, Chain, ScanChain, Frozen, Inverse
from .continuous import ContFlowDiffrax, ContFlowRK4
from .coupling import AffineCoupling, checker_mask
from .fourier import SpectrumScaling, FreeTheoryScaling, ToFourierData
from .linear import Scaling, Shift
from .meta import MetaLayer, ExpandDims, SqueezeDims, Reshape
from .onedim import *
from .splines import MonotoneRQSpline, apply_mrq_spline

__all__ = [
    # Base classes
    "Bijection",
    "Chain",
    "ScanChain",
    "Frozen",
    "Inverse",
    # Continuous flows
    "ContFlowDiffrax",
    "ContFlowRK4",
    # Coupling layers
    "AffineCoupling",
    "checker_mask",
    # Fourier-space bijections
    "SpectrumScaling",
    "FreeTheoryScaling",
    "ToFourierData",
    # Linear transformations
    "Scaling",
    "Shift",
    # Meta transformations
    "MetaLayer",
    "ExpandDims",
    "SqueezeDims",
    "Reshape",
    # Splines
    "MonotoneRQSpline",
    "apply_mrq_spline",
]
