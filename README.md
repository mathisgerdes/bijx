# (Lattice) normalizing flows with jax & flax

Updated and improved version of flow library using flax.nnx.

## Goals

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
lfx/
├── __init__.py                 # Main package exports (pulls from core, nn, sampling, etc.)
├── _version.py
├── utils.py                    # Truly generic, framework-agnostic utilities
├── sampling.py                 # Base distribution samplers, general sampling tools
│
├── core/                       # Core normalizing flow architectural components
│   ├── __init__.py             # Exports from this module (Bijection, specific flows, etc.)
│   ├── base.py                 # Abstract Bijection class, other base NF constructs
│   ├── transforms_1d.py        # Concrete 1D bijections (Affine, Splines, etc.)
│   ├── spectral.py             # Spectral domain bijections (SpectrumScaling)
│   ├── coupling.py             # Coupling layer architecture (uses bijections from transforms_1d)
│   ├── autoregressive.py       # Autoregressive flow architecture (uses bijections)
│   ├── # other_flow_arch.py    # E.g., for Planar, Glow-specific components if not in coupling
│   └── continuous/
│       ├── __init__.py
│       ├── base.py             # Base logic for continuous NFs (e.g., Neural ODE interface)
│       └── solvers.py          # ODE solvers (e.g., RK4)
│
├── nn/                         # Neural network building blocks (conditioners, etc.)
│   ├── __init__.py
│   ├── simple_nets.py
│   ├── conv.py
│   └── embeddings.py
│
└── applications/
    └── lattice/
        ├── __init__.py
        ├── fourier_ops.py      # LFT-specific Fourier ops (fft_momenta, masks)
        ├── bijections.py       # LFT-specific bijections (FreeTheoryScaling - uses core.spectral)
        ├── scalar_theory.py    # LFT scalar field theory models
        └── utils.py            # LFT-specific utilities
```
