"""
The `nn` package provides neural network components built on Flax NNX.

It is organized into the following submodules:

- **`conv`**: Implements specialized convolutions, such as those respecting lattice symmetries (`ConvSym`).
- **`embeddings`**: Contains modules that map low-dimensional inputs (typically scalars like time)
  into high-dimensional vector representations. Use these for conditioning on time or other scalar parameters.
- **`features`**: Contains modules that perform site-wise transformations on data tensors to create
  richer feature representations before they are mixed by other layers like convolutions.
- **`nets`**: Provides simple, fully-connected neural network architectures.
"""

from . import conv, embeddings, nets, features

__all__ = [
    "conv",
    "embeddings",
    "nets",
    "features",
]
