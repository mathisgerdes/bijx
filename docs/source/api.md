# API Reference

This section provides comprehensive documentation for all bijx components, organized by functionality.

## Core Components

The foundation of bijx consists of two main abstractions and their supporting infrastructure.

### Base Classes

```{eval-rst}
.. currentmodule:: bijx

.. autosummary::
   :toctree: _autosummary

   Bijection
   Distribution
   Sampler
```

## Bijections

Invertible transformations that form the building blocks of normalizing flows.

### Meta-Bijections

Composition and transformation utilities for building complex flows.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Chain
   ScanChain
   Inverse
   Frozen
```

### Linear Transformations

Simple but essential transformations for scaling, shifting, and reshaping.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Scaling
   Shift
   ExpandDims
   SqueezeDims
   Reshape
```

### One-Dimensional Bijections

Element-wise transformations that can be composed into coupling layers.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   OneDimensional
   BetaStretch
   GaussianCDF
   Sigmoid
   Tanh
   Tan
   Exponential
   SoftPlus
   Power
   AffineLinear
```

### Coupling Layers

Sophisticated transformations that update subsets of variables conditioned on others.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ModuleReconstructor
   GeneralCouplingLayer
   checker_mask
```

### Spline Transformations

Flexible piecewise transformations for smooth, expressive flows.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   MonotoneRQSpline
   rational_quadratic_spline
```

### Continuous Flows

ODE-based bijections for continuous-time dynamics.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ContFlowRK4
   ContFlowDiffrax
```

### Fourier-Space Transformations

Novel bijections operating in frequency domain for spatially-correlated systems.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ToFourierData
   SpectrumScaling
   FreeTheoryScaling
```

## Distributions

Probability distributions providing the foundation for normalizing flows.

### Basic Distributions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ArrayPrior
   IndependentNormal
   IndependentUniform
   DiagonalGMM
```

## Neural Networks

Building blocks for constructing neural networks within flows.

### Neural Network Modules

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   nn.simple_nets
   nn.conv
   nn.embeddings
```

## Lattice Field Theory

Specialized components for lattice field theory applications.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   lattice.scalar_vf
   lattice.scalartheory
```

## Fourier Utilities

Fourier transform utilities and representations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   fft_momenta
   FFTRep
   FourierData
   FourierMeta
```

## MCMC & Sampling

Markov Chain Monte Carlo samplers and utilities.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   IMH
   IMHState
   IMHInfo
   BufferedSampler
```

## ODE Solvers

Numerical integration for continuous normalizing flows.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ODESolver
   DiffraxConfig
   odeint_rk4
```

## Utilities

Helper functions and utilities for common operations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Const
   FrozenFilter
   ShapeInfo
   default_wrap
   effective_sample_size
   moving_average
   noise_model
   reverse_dkl
```
