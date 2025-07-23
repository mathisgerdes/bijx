from functools import partial

import chex
import flax.typing as ftp
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class Const(nnx.Variable):
    pass


# filter constants (above) and things wrapped in Frozen (defined below)
FrozenFilter = nnx.Any(Const, nnx.PathContains("frozen"))
ParamSpec = nnx.Variable | jax.Array | np.ndarray | chex.Shape | None


def load_shapes_magic():
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
    if not isinstance(x, tuple | list):
        return False
    return all(isinstance(dim, int) and dim > 0 for dim in x)


def default_wrap(
    x: ParamSpec,
    default=None,
    cls=nnx.Param,
    init_fn=nnx.initializers.normal(),
    init_cls=nnx.Param,
    rngs: nnx.Rngs | None = None,
):
    if x is None:
        x = default

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
def effective_sample_size(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Compute the ESS given log likelihoods.

    The two likelihood arrays must be evaluated for the same set of samples.
    The samples are assumed to be drawn from ``q``, such that ``logq``
    are the corresponding log-likelihoods, and ``p`` is the target.

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        The effective sample size per sample (between 0 and 1).
    """
    logw = logp - logq
    log_ess = 2 * jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2 * logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample


@jax.jit
def reverse_dkl(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Reverse KL divergence.

    Assuming likelihood arrays are evaluated for the same set of samples drawn
    from ``q``, this function then approximates the reverse KL divergence
    ``int_x q(x) log(q(x)/p(x)) dx``.

    If the samples were taken from p(x), the returned value is the negative
    forward KL divergence: ``int_x p(x) log(q(x)/p(x)) dx``.

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        Scalar representing the estimated reverse KL divergence.
    """
    return jnp.mean(logq - logp)


@partial(jax.jit, static_argnums=(1,))
def moving_average(x: jnp.ndarray, window: int = 10):
    """Moving average over 1d array."""
    if len(x) < window:
        return jnp.mean(x, keepdims=True)
    else:
        return jnp.convolve(x, jnp.ones(window), "valid") / window


def _none_or_tuple(x):
    return None if x is None else tuple(x)


class ShapeInfo:
    """
    Manages array shape information.

    Assumptions:
    - event_shape = space_shape + channel_shape
    - space_dim = len(space_shape)
    - channel_dim = len(channel_shape)

    Only partial information may be provided.
    Everything that can be derived will be computed.

    Args:
        event_shape: Full event shape (spatial + channel)
        space_shape: Spatial dimensions
        channel_shape: Channel dimensions
        space_dim: Number of spatial dimensions
        channel_dim: Number of channel dimensions
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

    @property
    def event_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.event_dim, 0))

    @property
    def channel_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.event_dim + self.space_dim, 0))

    @property
    def space_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.event_dim, -self.event_dim + self.space_dim))

    @property
    def event_size(self) -> int:
        return np.prod(self.event_shape, dtype=int)

    @property
    def space_size(self) -> int:
        return np.prod(self.space_shape, dtype=int)

    @property
    def channel_size(self) -> int:
        return np.prod(self.channel_shape, dtype=int)

    def tree_flatten(self):
        """Defines how to break down into JAX-compatible components"""
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
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

    def __hash__(self) -> int:
        _, info = self.tree_flatten()
        return hash(info)

    def __eq__(self, value: object) -> bool:
        _, info_self = self.tree_flatten()
        _, info_other = value.tree_flatten()
        return info_self == info_other


# Register as JAX pytree
jax.tree_util.register_pytree_node(
    ShapeInfo, ShapeInfo.tree_flatten, ShapeInfo.tree_unflatten
)


def noise_model(
    rng: nnx.Rngs | ftp.PRNGKey,
    model,
    scale=1,
    *filters,
    noise_fn=jax.random.normal,
):
    """Add noise to all model parameters (matching filters).

    This can be useful for testing purposes.
    """
    filter = nnx.Any(*filters) if filters else nnx.Param
    rngs = rng if isinstance(rng, nnx.Rngs) else nnx.Rngs(rng)

    graph, params, rest = nnx.split(model, filter, ...)
    params = jax.tree.map(
        lambda x: x + scale * noise_fn(rngs.sample(), x.shape),
        params,
    )
    return nnx.merge(graph, params, rest)
