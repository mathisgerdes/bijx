# Normalizing Flows with JAX/NNX

bijx is a normalizing flows library built with JAX and Flax.NNX, specifically designed for research in lattice field theory and machine learning. The library emphasizes flexibility and modularity over convenience, providing building blocks that can be composed rather than large monolithic classes.

## Key Features

- **JAX-First Design**: Built around JAX's functional programming paradigm with Flax.NNX for state management
- **Modular Architecture**: Small building blocks that compose together rather than large customizable classes
- **Flexible Parameter Specification**: Parameters can be provided as shapes, arrays, or existing `nnx.Variable` instances
- **Research-Focused**: Prioritizes flexibility and power over simple interfaces
- **Lattice Field Theory Integration**: Special focus on physics applications with Fourier space transformations

## Core Abstractions

Everything in bijx is built around two fundamental mathematical objects:

- **Bijections**: Invertible transformations that track how they affect probability densities
- **Distributions**: Probability distributions with sampling and density evaluation methods

## Quick Start

```python
import jax.numpy as jnp
from flax import nnx
import bijx

# Create RNG keys
rngs = nnx.Rngs(0)

# Define a simple normalizing flow
prior = bijx.IndependentNormal((2,), rngs=rngs)
bijection = bijx.Chain(
    bijx.Scaling(jnp.array([2.0, 1.0])),
    bijx.TanhLayer(),
)
flow = bijx.Transformed(prior, bijection)

# Sample from the flow
samples, log_densities = flow.sample(batch_shape=(1000,))
```


## Installation

```bash
# Install from source
git clone https://github.com/mathisgerdes/bijx.git
cd bijx
pip install -e .

# With development & testing dependencies
pip install -e ".[dev]"
```

## Citation

To be added...

## Contents

```{toctree}
:maxdepth: 2

intro/concepts
intro/basic
intro/coupling
intro/continuous
intro/flowjax
intro/fourier
intro/lie
intro/cg

tutorials
api
```
