"""Pytest configuration for doctest support."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Configure JAX for reproducible tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """Automatically add common imports and objects to all doctests."""
    import bijx

    # Create common test objects
    rng = nnx.Rngs(42)

    # Mock classes for examples that need placeholders
    class SomeBijection(bijx.Bijection):
        def forward(self, x, log_density, **kwargs):
            return x, log_density

        def reverse(self, x, log_density, **kwargs):
            return x, log_density

    class SomeArrayDistribution(bijx.ArrayDistribution):
        def __init__(self, event_shape, rngs=None):
            super().__init__(event_shape, rngs=rngs or nnx.Rngs(42))

        def sample(self, batch_shape=(), rng=None, **kwargs):
            key = jax.random.PRNGKey(42)
            shape = batch_shape + self.event_shape
            x = jax.random.normal(key, shape)
            log_p = jnp.zeros(batch_shape)
            return x, log_p

        def log_density(self, x, **kwargs):
            batch_shape = x.shape[: len(x.shape) - len(self.event_shape)]
            return jnp.zeros(batch_shape)

    # Create a simple model for noise_model example
    class SimpleModel(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(10, 1, rngs=rngs)

    model = SimpleModel(rng)

    # Add everything to doctest namespace
    doctest_namespace.update(
        {
            # Core imports
            "bijx": bijx,
            "jax": jax,
            "jnp": jnp,
            "nnx": nnx,
            # Common objects
            "rng": rng,
            "rngs": rng,  # alias for compatibility
            "model": model,
            # Mock classes
            "SomeBijection": SomeBijection,
            "SomeArrayDistribution": SomeArrayDistribution,
        }
    )
