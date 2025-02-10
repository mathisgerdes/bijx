from functools import partial

import flax
import jax
import jax.numpy as jnp


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
    log_ess = 2*jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2*logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample


@partial(jax.jit, static_argnums=(1,))
def moving_average(x: jnp.ndarray, window: int = 10):
    """Moving average over 1d array."""
    if len(x) < window:
        return jnp.mean(x, keepdims=True)
    else:
        return jnp.convolve(x, jnp.ones(window), 'valid') / window


def _none_or_tuple(x):
    return None if x is None else tuple(x)


@flax.struct.dataclass
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

    @classmethod
    def init(
            cls,
            *,
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
                channel_shape = event_shape[-channel_dim:]

        return cls(
            event_shape=event_shape,
            space_shape=space_shape,
            channel_shape=channel_shape,
            event_dim=event_dim,
            space_dim=space_dim,
            channel_dim=channel_dim,
        )

    def process_event(self, batched_shape: tuple[int, ...]):
        if self.event_dim is None:
            raise RuntimeError("event dimension is unknown; cannot process event")

        batch_shape = batched_shape[:-self.event_dim]
        event_shape = batched_shape[-self.event_dim:]
        assert self.event_shape is None or self.event_shape == event_shape, \
            f"event shape mismatch: {self.event_shape=} != {event_shape=}"

        return batch_shape, self.init(
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
