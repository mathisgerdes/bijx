<p align="center">
    <img src="docs/source/_static/icons/bijx.svg" alt="bijx logo" height="100">
    <br>
    <em><b>bij</b>ection + ja<b>x</b> = /ˈbaɪdʒæks/</em>
</p>

# Bijections & normalizing flows with JAX/NNX

This library aims to provide flexible tools for building normalizing flows and doing research.
In particular, the focus is not to expose a simple and safe interface of the most common use cases.
Rather, the aim is to provide flexibility and powerful reusable building blocks.

The library is built around two fundamental mathematical objects:

- **Bijections**: Invertible transformations that track how they affect probability densities
- **Distributions**: Probability distributions with sampling and density evaluation methods

Some implementations are tailored specifically to applications in physics and especially lattice field theory.

## Design principles

- Expose as many of the building blocks as possible, in some cases provide multiple ways to achieve the same thing for convenience.
- Modularity: any part of the library should be usable on its own.
- Err on the side of more flexibility, sometimes at the cost of more possible ways break things.
- Use run-time shape inference exploiting some basic shape assumptions (`batch + space + channels`, where `event_shape = space + channels`) and [auto_vmap](https://github.com/mathisgerdes/jax_autovmap) for flexibility.

## Installation

```bash
pip install -e .
```

For development and testing, install as editable package with all dependencies:

```bash
pip install -e ".[dev]"
```


## Documentation

To compile and open local server, run `make livehtml` in the docs directory.

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
