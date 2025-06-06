from .nn import conv
from .core import bijections, discrete, fourier, rk4ode
from . import lattice, sampling, utils
from ._version import __version__
from .core.bijections import (
    Bijection,
    Chain,
    ScanChain,
    ExpandDims,
    Frozen,
    Inverse,
    MetaLayer,
    Reshape,
    Scaling,
    Shift,
    SqueezeDims,
)
from .nn.conv import ConvSym, kernel_d4, kernel_equidist
from .core.discrete import (
    AffineCoupling,
    MonotoneRQSpline,
    apply_mrq_spline,
    checker_mask,
)
from .nn.embeddings import KernelFourier, KernelGauss, KernelLin, KernelReduced
from .core.fourier import FreeTheoryScaling, SpectrumScaling, fft_momenta
from .core.ode import ODESolver, DiffraxConfig, ContFlowRK4, ContFlowDiffrax
from .core.one_dim import (
    BetaStretch,
    GaussianCDF,
    SigmoidLayer,
    TanhLayer,
    TanLayer,
    OneDimensional,
)
from .core.rk4ode import odeint_rk4
from .sampling import (
    ArrayPrior,
    BufferedSampler,
    IndependentNormal,
    IndependentUniform,
    Distribution,
    Sampler,
)
from .lattice.scalar_vf import (
    ConcatFeatures,
    FourierFeatures,
    NonlinearFeatures,
    Phi4CNF,
    PolynomialFeatures,
)
from .nn.simple_nets import SimpleConvNet, SimpleResNet
from .utils import (
    Const,
    FrozenFilter,
    ShapeInfo,
    default_wrap,
    effective_sample_size,
    moving_average,
    noise_model,
    reverse_dkl,
)

__all__ = [
    # Core classes
    "Bijection",
    "Distribution",
    "Sampler",
    # Bijection classes
    "Chain",
    "ScanChain",
    "ExpandDims",
    "Frozen",
    "Inverse",
    "MetaLayer",
    "Reshape",
    "Scaling",
    "Shift",
    "SqueezeDims",
    # Convolution
    "ConvSym",
    "kernel_d4",
    "kernel_equidist",
    # Discrete bijections
    "AffineCoupling",
    "MonotoneRQSpline",
    "apply_mrq_spline",
    "checker_mask",
    # Embeddings
    "KernelFourier",
    "KernelGauss",
    "KernelLin",
    "KernelReduced",
    # Fourier transforms
    "FreeTheoryScaling",
    "SpectrumScaling",
    "fft_momenta",
    # ODE solvers
    "ODESolver",
    "DiffraxConfig",
    "ContFlowRK4",
    "ContFlowDiffrax",
    # One-dimensional transforms
    "BetaStretch",
    "GaussianCDF",
    "SigmoidLayer",
    "TanhLayer",
    "TanLayer",
    "OneDimensional",
    # RK4 ODE integration
    "odeint_rk4",
    # Sampling/distributions
    "ArrayPrior",
    "BufferedSampler",
    "IndependentNormal",
    "IndependentUniform",
    # Lattice field features
    "ConcatFeatures",
    "FourierFeatures",
    "NonlinearFeatures",
    "Phi4CNF",
    "PolynomialFeatures",
    # Neural networks
    "SimpleConvNet",
    "SimpleResNet",
    # Utilities
    "Const",
    "FrozenFilter",
    "ShapeInfo",
    "default_wrap",
    "effective_sample_size",
    "moving_average",
    "noise_model",
    "reverse_dkl",
]
