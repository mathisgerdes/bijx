# Normalizing flows with jax & flax

Updated and improved version of flow library using flax.nnx.

## Goals

Generally, this library aims to provide flexible tools for building normalizing flows and doing research.
In particular, the focus is not to expose a simple and safe interface.
Rather, as much power and flexibility is given to the user.

- Expose as many of the building blocks to the user as possible.
- Sacrifice some convenience for modularity (i.e. repeat some code, but then easier to implement new ideas).
- Using one part of the library should not force you to use the rest (as much as possible).
- Instead of providing large general and customizable classes/functions, aim to provide small building blocks that can be composed together.

## Installation

```bash
pip install -e .
```

## Documentation

To compile and open local server, run `make livehtml` in the docs directory.

# Basic design principles and notes

## Parameter Specification

The library aims for flexibility in how layer parameters (e.g., scales, shifts) are specified, especially for basic building-block bijections.
When constructing a layer, parameters can typically be provided in several ways:
1.  As a shape tuple (e.g., `(D,)` or `()`). In this case, a new `flax.nnx.Param` will be created and initialized according to the layer's default initializer.
2.  As a Jax or NumPy array. This value will be wrapped into a new `flax.nnx.Param`, using the provided array as its initial value. Depending on context, other subtypes of `flax.nnx.Variable` may be used.
3.  As an existing `flax.nnx.Variable` (e.g., `flax.nnx.Param` or a custom `utils.Const`). This allows for sharing parameters between layers or using non-trainable constants.

This approach, facilitated by the `utils.ParamSpec` type hint and `utils.default_wrap` function, provides a convenient shorthand for common cases (initialization by shape or value) while still allowing fine-grained control when needed (using existing variables).

## Samplers and random keys

- Samples hold an `nnx.Rngs` instance, which is used if no manual `rng` is provided to the sampling methods.
- Generally not attempting to replace MCMC packages like [blackjax](https://blackjax-devs.github.io/blackjax/). Still providing two tools for convenience:
    - An independent Metropolis-Hastings sampler, since our methods samples and likelihoods are typically generated at the same time.
    - A batched sampler.

## Shapes and inputs

- Generally inputs must be arrays or pytrees of arrays. floats/ints are not always supported.
- Batch dimensions are generally the first axes, followed by "space" and possibly "channel" dimensions, depending on context. Everything that is not a batch dimension is considered an "event" dimension.
- Many methods rely on `x` (i.e. the event) and `log_density` to have the same batch dimensions to infer the "event" shape. This should _always_ be true.
- In addition, all distributions/priors must implement `get_batch_shape(x)`, which returns the batch shape of the distribution. The reason not to directly implement `event_shape` is to support more general pytrees as "events".

## Module layout

```
bijx/
├── __init__.py                 # Main package exports (simplified structure)
├── utils.py                    # Utilities
├── distributions.py
├── fourier.py
├── mcmc.py
├── samplers.py
├── solvers.py
│
├── bijections/                 # All bijection-related code
│   ├── __init__.py
│   ├── base.py                 # Bijection, Chain, ...
│   ├── coupling.py             # Coupling layers and utilities
│   ├── fourier.py              # SpectrumScaling, FreeTheoryScaling, ToFourierData
│   ├── linear.py               # Scaling, Shift
│   ├── meta.py                 # Reshape, ExpandDims
│   ├── onedim.py               # GaussianCDF, TanhLayer, etc.
│   ├── splines.py              # Spline layers
│   └── continuous.py           # Continuous flow solvers
│
├── nn/
│   ├── __init__.py
│   ├── simple_nets.py
│   ├── conv.py
│   └── embeddings.py
│
└── lattice/                    # Lattice field theory applications
    ├── __init__.py
    ├── scalar_vf.py
    └── scalartheory.py
```
