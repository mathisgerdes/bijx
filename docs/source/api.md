# API Reference

This section provides comprehensive documentation for all bijx components, organized by functionality.

## Main components

```{eval-rst}
.. automodule:: bijx

.. currentmodule:: bijx
```

### Core Classes and Base Types
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Bijection
   Distribution
   ArrayDistribution
   ApplyBijection
```

### Distributions
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   IndependentNormal
   IndependentUniform
   MultivariateNormal
   DiagonalNormal
   MixtureStack
   GaussianMixture
```

### Sampling and Transforms
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Transformed
   BufferedSampler
```

### Bijection Composition and Meta-bijections
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Chain
   ScanChain
   Inverse
   CondInverse
   Frozen
```

"Meta" bijections that do not change the log-density.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   MetaLayer
   ExpandDims
   SqueezeDims
   Reshape
   Partial
```

### General Coupling and Masking
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   GeneralCouplingLayer
   BinaryMask
   checker_mask
   ModuleReconstructor
```

### Spline Bijections
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   MonotoneRQSpline
   rational_quadratic_spline
```

### Continuous Flows
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ContFlowCG
   ContFlowDiffrax
   ContFlowRK4
   ConvVF
   AutoJacVF
```

### One-dimensional Bijections
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ScalarBijection
   AffineLinear
   Scaling
   Shift
   BetaStretch
   Exponential
   GaussianCDF
   Power
   Sigmoid
   Sinh
   SoftPlus
   Tan
   Tanh
```

### Fourier and Physics-specific Bijections
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ToFourierData
   FreeTheoryScaling
   SpectrumScaling
```

### ODE Solvers
```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   DiffraxConfig
   odeint_rk4
```

### MCMC

These tools mimic the API of [blackjax](https://blackjax-devs.github.io/blackjax/),
with the main difference that the samples and the proposal densities are generated simultaneously.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   IMH
   IMHState
   IMHInfo
```

### Utilities
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
   load_shapes_magic
```

## Submodules

Core submodules provide tools for lattice field theory, Fourier transformations, and more.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   fourier
   lie
   cg
   lattice
   lattice.gauge
   lattice.scalar
```

For interfacing with [flowjax](https://github.com/danielward27/flowjax),the following submodule can be used.
Since flowjax is not an explicit dependency of bijx, it has to be imported explicitly.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   flowjax
```

`bijx.nn` provides building blocks for neural networks and prototyping.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   nn.conv
   nn.embeddings
   nn.features
   nn.nets
```
