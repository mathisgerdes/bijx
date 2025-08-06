r"""Bijx: Bijections and normalizing flows with JAX, with focus on physics.

Bijx is a library for normalizing flows built on JAX and Flax NNX,
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

from . import (
    bijections,
    cg,
    distributions,
    fourier,
    lattice,
    lie,
    mcmc,
    nn,
    samplers,
    solvers,
    utils,
)
from ._version import __version__
from .bijections import (
    AffineLinear,
    ApplyBijection,
    AutoJacVF,
    Bijection,
    BinaryMask,
    Chain,
    ContFlowDiffrax,
    ContFlowCG,
    ContFlowRK4,
    ConvCNF,
    ExpandDims,
    FreeTheoryScaling,
    Frozen,
    GeneralCouplingLayer,
    Inverse,
    CondInverse,
    MetaLayer,
    MonotoneRQSpline,
    ModuleReconstructor,
    Reshape,
    ScalarBijection,
    Scaling,
    ScanChain,
    Shift,
    SpectrumScaling,
    SqueezeDims,
    ToFourierData,
    checker_mask,
    rational_quadratic_spline,
)
from .bijections.scalar import (
    BetaStretch,
    Exponential,
    GaussianCDF,
    Power,
    Sigmoid,
    Sinh,
    SoftPlus,
    Tan,
    Tanh,
)
from .distributions import (
    ArrayDistribution,
    DiagonalGMM,
    Distribution,
    IndependentNormal,
    IndependentUniform,
)
from .fourier import FFTRep, FourierData, FourierMeta, fft_momenta
from .mcmc import IMH, IMHInfo, IMHState
from .nn import conv, embeddings, features, nets
from .nn.conv import ConvSym, kernel_d4, kernel_equidist
from .nn.embeddings import (
    KernelFourier,
    KernelGauss,
    KernelLin,
    KernelReduced,
    PositionalEmbedding,
)
from .nn.features import (
    ConcatFeatures,
    FourierFeatures,
    NonlinearFeatures,
    PolynomialFeatures,
)
from .samplers import BufferedSampler, Transformed
from .solvers import DiffraxConfig, odeint_rk4
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
    # Submodules
    "bijections",
    "distributions",
    "fourier",
    "lattice",
    "mcmc",
    "nn",
    "samplers",
    "solvers",
    "utils",
    "lie",
    "cg",
    "features",
    # Core classes
    "Bijection",
    "Distribution",
    "Transformed",
    # Bijection classes
    "ApplyBijection",
    "Chain",
    "ScanChain",
    "ExpandDims",
    "Frozen",
    "Inverse",
    "CondInverse",
    "MetaLayer",
    "Reshape",
    "ToFourierData",
    "Scaling",
    "Shift",
    "SqueezeDims",
    "AutoJacVF",
    "ContFlowCG",
    "ConvCNF",
    # Convolution
    "ConvSym",
    "kernel_d4",
    "kernel_equidist",
    # Discrete bijections
    "ModuleReconstructor",
    "BinaryMask",
    "GeneralCouplingLayer",
    "MonotoneRQSpline",
    "rational_quadratic_spline",
    "checker_mask",
    # Distributions
    "ArrayDistribution",
    "IndependentNormal",
    "IndependentUniform",
    "DiagonalGMM",
    # Embeddings
    "KernelFourier",
    "KernelGauss",
    "KernelLin",
    "KernelReduced",
    "PositionalEmbedding",
    # Fourier transforms
    "FreeTheoryScaling",
    "SpectrumScaling",
    "fft_momenta",
    "FFTRep",
    "FourierData",
    "FourierMeta",
    # MCMC
    "IMH",
    "IMHState",
    "IMHInfo",
    # ODE solvers
    "DiffraxConfig",
    "ContFlowRK4",
    "ContFlowDiffrax",
    "odeint_rk4",
    # One-dimensional transforms
    "BetaStretch",
    "GaussianCDF",
    "Sigmoid",
    "Sinh",
    "Tanh",
    "Tan",
    "Exponential",
    "SoftPlus",
    "Power",
    "AffineLinear",
    "Scaling",
    "Shift",
    "ScalarBijection",
    # Sampling
    "BufferedSampler",
    # Features
    "ConcatFeatures",
    "FourierFeatures",
    "NonlinearFeatures",
    "PolynomialFeatures",
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
