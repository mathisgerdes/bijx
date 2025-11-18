"""
Utility functions and classes for bijx library.

This module provides essential utilities for parameter handling, shape management,
and common operations used throughout the bijx library. Key components include
parameter specification systems, shape inference utilities, and statistical metrics.
"""

from functools import partial

import chex
import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = [
    "Const",
    "FrozenFilter",
    "ShapeInfo",
    "default_wrap",
    "effective_sample_size",
    "moving_average",
    "noise_model",
    "reverse_dkl",
    "load_shapes_magic",
]


class Const(nnx.Variable):
    """Mark a variable as constant during training.

    This variable type explicitly indicates that a parameter should remain
    fixed during optimization. Useful for freezing parts of models or
    storing hyperparameters that should not be updated.

    Can be used with :obj:`FrozenFilter` to selectively freeze parameters
    during gradient-based optimization.

    Example:
        >>> bijection = SomeBijection()
        >>> bijection.scale = Const(1.0)  # Fix scale parameter
        >>> # standard optimizers will not apply gradients to scale
    """

    pass


FrozenFilter = nnx.Any(Const, nnx.PathContains("frozen"))
# Explicitly set the docstring for Sphinx autodoc
FrozenFilter.__doc__ = """
Filter that matches constant variables and anything in a 'frozen' path.

Used to identify parameters that should not be updated during training.
Matches both Const variables and any parameters with 'frozen' in their path.
"""

ParamSpec = nnx.Variable | jax.Array | np.ndarray | chex.Shape
"""Type specification for flexible parameter initialization.

Accepts:
- nnx.Variable: Already wrapped parameter
- Arrays: Direct parameter values
- Shapes: Initialize with given shape using init_fn
"""


def load_shapes_magic():
    """Load IPython magic command for inspecting JAX pytree shapes.

    Registers the `%shapes` magic command that displays the shape structure
    of any JAX pytree. Useful for debugging tensor dimensions in notebooks.

    # In IPython/Jupyter:
    # %shapes some_complex_pytree_or_expression
    # Displays nested shape structure

    Note:
        Only works in IPython/Jupyter environments. Prints warning if
        IPython is not available.
    """
    try:
        from IPython import get_ipython
        from IPython.core.magic import Magics, line_magic, magics_class

        ip = get_ipython()

        @magics_class
        class ShapesMagic(Magics):
            @line_magic
            def shapes(self, line):
                output = eval(line, self.shell.user_ns)
                print(jax.tree.map(jnp.shape, output))

        ip.register_magics(ShapesMagic)
    except ImportError:
        print("Warning: IPython not found; shapes magic not loaded")


def is_shape(x):
    """Check if object represents a valid array shape.

    Args:
        x: Object to test.

    Returns:
        True if x is a tuple/list of positive integers.
    """
    if not isinstance(x, tuple | list):
        return False
    return all(isinstance(dim, int) and dim > 0 for dim in x)


def default_wrap(
    x: ParamSpec,
    cls=nnx.Param,
    init_fn=nnx.initializers.normal(),
    init_cls=nnx.Param,
    rngs: nnx.Rngs | None = None,
):
    """Flexibly wrap parameter specifications into nnx.Variable instances.

    This function provides a unified interface for parameter initialization,
    accepting various input types and converting them to appropriate
    nnx.Variable instances for use in Flax modules.

    Args:
        x: Parameter specification (Variable, array, or shape).
        cls: Variable class to use for array wrapping.
        init_fn: Initialization function for shape-based parameters.
        init_cls: Variable class to use for initialized parameters.
        rngs: Random number generators for initialization.

    Returns:
        Wrapped parameter as an nnx.Variable instance.

    Raises:
        ValueError: If parameter specification type is not supported.

    Example:
        >>> # Direct array
        >>> param = default_wrap(jnp.array([1.0, 2.0]))
        >>> # Shape-based initialization
        >>> param = default_wrap((10, 5), rngs=rngs)
        >>> # Already wrapped
        >>> some_array = jnp.array([1.0, 2.0])
        >>> param = default_wrap(nnx.Param(some_array))
    """
    if isinstance(x, nnx.Variable):
        return x
    elif isinstance(x, jax.Array | np.ndarray):
        return cls(jnp.asarray(x))
    elif is_shape(x):
        return init_cls(init_fn(rngs.params(), x))
    else:
        raise ValueError(
            f"Cannot process parameter specification of type {type(x)}: {x}"
        )


@jax.jit
def effective_sample_size(
    target_ld: jnp.ndarray, sample_ld: jnp.ndarray
) -> jnp.ndarray:
    r"""Compute effective sample size from importance weights.

    Measures the efficiency of importance sampling by computing the effective
    number of independent samples. Values close to 1 indicate efficient sampling,
    while values near 0 suggest poor importance weight distribution.

    Uses the formula: $\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}$
    where importance weights are $w_i = \frac{p(x_i)}{q(x_i)}$.

    Args:
        target_ld: Log likelihood of target distribution p (up to constant).
        sample_ld: Log likelihood of proposal distribution q (up to constant).

    Returns:
        Effective sample size per sample (between 0 and 1).

    Note:
        Input arrays must correspond to the same set of samples drawn from q.
        Uses numerically stable log-sum-exp computations.
    """
    logw = target_ld - sample_ld
    log_ess = 2 * jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2 * logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample


@jax.jit
def reverse_dkl(target_ld: jnp.ndarray, sample_ld: jnp.ndarray) -> jnp.ndarray:
    r"""Estimate reverse Kullback-Leibler divergence.

    Computes an importance sampling estimate of the reverse KL divergence
    $D_{\text{KL}}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx$.

    Here, $q$ is ``sample_ld`` and $p$ is ``target_ld``.

    When samples are drawn from q, this gives the reverse KL.
    If samples are drawn from p, this gives the negative forward KL divergence.

    Args:
        target_ld: Log likelihood of distribution p (up to constant shift).
        sample_ld: Log likelihood of distribution q (up to constant shift).

    Returns:
        Estimated reverse KL divergence as a scalar.

    Note:
        Input arrays must correspond to the same sample set. The sampling
        distribution determines which KL direction is being estimated.
    """
    return jnp.mean(sample_ld - target_ld)


@partial(jax.jit, static_argnums=(1,))
def moving_average(x: jnp.ndarray, window: int = 10):
    """Compute moving average of a 1D array.

    Applies a sliding window average to smooth noisy signals or time series.
    For arrays shorter than the window size, returns the overall mean.

    Args:
        x: Input 1D array to smooth.
        window: Size of the moving average window.

    Returns:
        Smoothed array with length max(1, len(x) - window + 1).

    Example:
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> smoothed = moving_average(x, window=3)
        >>> # Returns [2.0, 3.0, 4.0] (averages of [1,2,3], [2,3,4], [3,4,5])
    """
    if len(x) < window:
        return jnp.mean(x, keepdims=True)
    else:
        return jnp.convolve(x, jnp.ones(window), "valid") / window


def _none_or_tuple(x):
    """Convert input to tuple or return None if input is None."""
    return None if x is None else tuple(x)


class ShapeInfo:
    """Comprehensive shape information manager for array operations.

    Manages the relationship between different types of array dimensions:
    event, spatial, channel, and batch dimensions. Automatically infers
    missing information from provided specifications.

    The shape hierarchy follows: event_shape = space_shape + channel_shape
    This enables structured operations on multi-dimensional data like images,
    lattice fields, or other spatially-structured data with channels.

    Args:
        event_shape: Complete event dimensions (spatial + channel).
        space_shape: Spatial dimensions only.
        channel_shape: Channel dimensions only.
        event_dim: Number of event dimensions.
        space_dim: Number of spatial dimensions.
        channel_dim: Number of channel dimensions.

    Note:
        Only partial information needs to be provided. Missing information
        is automatically inferred from the provided specifications.

    Example:
        >>> # For 2D images with RGB channels
        >>> info = ShapeInfo(event_shape=(32, 32, 3), space_dim=2)
        >>> info.space_shape  # Can be set via space_dim=2
        (32, 32)
        >>> info.channel_shape  # Can be set via channel_dim=1
        (3,)
    """

    event_shape: tuple[int, ...] | None = None
    space_shape: tuple[int, ...] | None = None
    channel_shape: tuple[int, ...] | None = None
    event_dim: int | None = None
    space_dim: int | None = None
    channel_dim: int | None = None

    def __init__(
        self,
        event_shape: tuple[int, ...] | None = None,
        space_shape: tuple[int, ...] | None = None,
        channel_shape: tuple[int, ...] | None = None,
        event_dim: int | None = None,
        channel_dim: int | None = None,
        space_dim: int | None = None,
    ):

        event_shape = _none_or_tuple(event_shape)
        space_shape = _none_or_tuple(space_shape)
        channel_shape = _none_or_tuple(channel_shape)

        # space + channel -> event
        if space_shape is not None and channel_shape is not None:
            event_shape = space_shape + channel_shape

        # shapes -> dims
        if space_shape is not None:
            space_dim = len(space_shape)
        if channel_shape is not None:
            channel_dim = len(channel_shape)
        if event_shape is not None:
            event_dim = len(event_shape)

        # dim -> dim
        if space_dim is not None and channel_dim is not None:
            event_dim = space_dim + channel_dim
        else:
            if event_dim is not None and space_dim is not None:
                channel_dim = event_dim - space_dim
            if event_dim is not None and channel_dim is not None:
                space_dim = event_dim - channel_dim

        # event + dims -> space/channel
        if event_shape is not None:
            if space_dim is not None:
                space_shape = event_shape[:space_dim]
            if channel_dim is not None:
                channel_shape = () if channel_dim == 0 else event_shape[-channel_dim:]

        self.event_shape = event_shape
        self.space_shape = space_shape
        self.channel_shape = channel_shape
        self.event_dim = event_dim
        self.space_dim = space_dim
        self.channel_dim = channel_dim

    def process_event(self, batched_shape: tuple[int, ...]):
        """Separate batch and event dimensions from a complete shape.

        Also infers event, channel, space shapes, if only dimensions were known.

        Args:
            batched_shape: Complete shape including batch and event dimensions.

        Returns:
            Tuple of (batch_shape, event_shape_info) where event_shape_info
            is a new ShapeInfo instance with inferred event dimensions.

        Raises:
            RuntimeError: If event dimension is unknown.
            AssertionError: If provided event_shape doesn't match inferred shape.
        """
        if self.event_dim is None:
            raise RuntimeError("event dimension is unknown; cannot process event")

        if self.event_dim == 0:
            return batched_shape, ShapeInfo(
                event_shape=(),
                space_shape=(),
                channel_shape=(),
            )

        batch_shape = batched_shape[: -self.event_dim]
        event_shape = batched_shape[-self.event_dim :]
        assert (
            self.event_shape is None or self.event_shape == event_shape
        ), f"event shape mismatch: {self.event_shape=} != {event_shape=}"

        return batch_shape, ShapeInfo(
            event_shape=event_shape,
            space_shape=self.space_shape,
            channel_shape=self.channel_shape,
            event_dim=self.event_dim,
            space_dim=self.space_dim,
            channel_dim=self.channel_dim,
        )

    def process_and_flatten(self, x: jax.Array):
        """Process array and flatten event dimensions.

        Args:
            x: Input array with batch and event dimensions.

        Returns:
            Tuple of (flattened_array, batch_shape, shape_info) where event dimensions
            are flattened into a single dimension and batch shape is preserved.
        """
        batched_shape = jnp.shape(x)
        batch_shape, info = self.process_event(batched_shape)

        return jnp.reshape(x, batch_shape + (-1,)), batch_shape, info

    def process_and_canonicalize(self, x: jax.Array):
        """Process array and canonicalize event dimensions to (batch, *space, channel).

        Args:
            x: Input array with batch and event dimensions.

        Returns:
            Tuple of (canonicalized_array, batch_shape, shape_info)
            where event dimensions are canonicalized into a single batch and single
            channel dimension, preserving the space dimensions.
        """
        batched_shape = jnp.shape(x)
        batch_shape, info = self.process_event(batched_shape)
        batch_size = np.prod(batch_shape, dtype=int)
        canonical = jnp.reshape(x, (batch_size, *info.space_shape, info.channel_size))
        return canonical, batch_shape, info

    @property
    def event_axes(self) -> tuple[int, ...]:
        """Axis indices for event dimensions."""
        return tuple(range(-self.event_dim, 0))

    @property
    def channel_axes(self) -> tuple[int, ...]:
        """Axis indices for channel dimensions."""
        return tuple(range(-self.event_dim + self.space_dim, 0))

    @property
    def space_axes(self) -> tuple[int, ...]:
        """Axis indices for spatial dimensions."""
        return tuple(range(-self.event_dim, -self.event_dim + self.space_dim))

    @property
    def event_size(self) -> int:
        """Total number of elements in event shape."""
        return np.prod(self.event_shape, dtype=int)

    @property
    def space_size(self) -> int:
        """Total number of elements in spatial dimensions."""
        return np.prod(self.space_shape, dtype=int)

    @property
    def channel_size(self) -> int:
        """Total number of elements in channel dimensions."""
        return np.prod(self.channel_shape, dtype=int)

    def _tree_flatten(self):
        """JAX pytree flattening for compatibility with transformations."""
        children = ()  # No array leaves in this class
        aux_data = {
            "event_shape": self.event_shape,
            "space_shape": self.space_shape,
            "channel_shape": self.channel_shape,
            "event_dim": self.event_dim,
            "space_dim": self.space_dim,
            "channel_dim": self.channel_dim,
        }
        return children, aux_data

    def __repr__(self):
        attrs = [
            f"event_shape={self.event_shape}",
            f"space_shape={self.space_shape}",
            f"channel_shape={self.channel_shape}",
            f"event_dim={self.event_dim}",
            f"space_dim={self.space_dim}",
            f"channel_dim={self.channel_dim}",
        ]
        return f"ShapeInfo({', '.join(attrs)})"

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def __hash__(self) -> int:
        _, info = self._tree_flatten()
        return hash(info)

    def __eq__(self, value: object) -> bool:
        _, info_self = self._tree_flatten()
        _, info_other = value._tree_flatten()
        return info_self == info_other


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    ShapeInfo, ShapeInfo._tree_flatten, ShapeInfo._tree_unflatten
)


def noise_model(
    rng: nnx.Rngs | ftp.PRNGKey,
    model,
    scale=1,
    *filters,
    noise_fn=jax.random.normal,
):
    """Add random noise to model parameters for testing or regularization.

    Applies additive noise to all parameters matching the specified filters.
    Useful for testing model robustness, implementing parameter regularization,
    or studying sensitivity to parameter perturbations.

    Args:
        rng: Random number generator or key for noise generation.
        model: Flax NNX model to add noise to.
        scale: Scaling factor for noise magnitude.
        *filters: Parameter filters to select which parameters to perturb.
        noise_fn: Function for generating noise (default: normal distribution).

    Returns:
        New model instance with noisy parameters.

    Example:
        >>> noisy_model = noise_model(rng, model, scale=0.1)
        >>> # Adds Gaussian noise with std=0.1 to all parameters

    Note:
        If no filters are provided, defaults to nnx.Param.
    """
    filter = nnx.Any(*filters) if filters else nnx.Param
    rngs = rng if isinstance(rng, nnx.Rngs) else nnx.Rngs(rng)

    graph, params, rest = nnx.split(model, filter, ...)
    params = jax.tree.map(
        lambda x: x + scale * noise_fn(rngs.sample(), x.shape),
        params,
    )
    return nnx.merge(graph, params, rest)
