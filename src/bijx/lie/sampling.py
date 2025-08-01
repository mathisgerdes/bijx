import jax
import jax.numpy as jnp
import numpy as np


@jax.vmap
def _sample_haar(z):
    # if this is a bottleneck, investigate https://github.com/google/jax/issues/8542
    q, r = jnp.linalg.qr(z)
    d = jnp.diag(r)
    d = d / jnp.abs(d)
    norm = jnp.prod(d) * jnp.linalg.det(q)
    m = jnp.einsum('ij,j->ij', q, d / norm**(1/len(d)))
    return m


def sample_haar(rng, n, count):
    """Sample SU(N) matrices uniformly according to Haar measure."""
    real_part, imag_part = 1/np.sqrt(2) * jax.random.normal(rng, (2, count, n, n))
    z = real_part + 1j * imag_part
    return _sample_haar(z)


def sample_haar_lattice(rng, count, shape, n=2):
    dim = len(shape)
    size = count * np.prod(shape) * dim
    lat = sample_haar(rng, n, size).reshape(count, *shape, dim, n, n)
    return lat
