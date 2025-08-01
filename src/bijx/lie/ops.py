import jax.numpy as jnp
from einops import einsum


def contract(*factors, trace=False, return_einsum_indices=False):
    """Contrast chain of matrices.

    Except for last two axes (which are contracted), broadcast
    factors from left to right (opposite of numpy).
    """
    leading = [jnp.ndim(f) - 2 for f in factors]
    assert all(l >= 0 for l in leading), 'all factors must be matrices (ndim >= 2)'

    indices = []
    for m, l in enumerate(leading):
        indices.append([f'l{i}' for i in range(l)] + [f'm{m}', f'm{m + 1}'])

    if trace:
        indices[-1][-1] = 'm0'

    ind_in = ', '.join(' '.join(ind) for ind in indices)
    ind_out = ' '.join(f'l{i}' for i in range(max(leading)))
    if not trace:
        ind_out += f' m0 m{len(factors)}'
    if return_einsum_indices:
        return ind_in, ind_out
    return einsum(*factors, f'{ind_in} -> {ind_out}')


def scalar_prod(a, b):
    """Compute scalar product between lie algebra elements a & b."""
    return jnp.einsum('...ij,...ij', a.conj(), b) / 2


def adjoint(arr):
    return arr.conj().swapaxes(-1, -2)
