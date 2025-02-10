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

- [Tutorial: Basics](tutorial-basics.ipynb)
- [Tutorial: Free Theory](tutorial-phi4.ipynb)
