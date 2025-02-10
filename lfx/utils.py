from functools import partial

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
