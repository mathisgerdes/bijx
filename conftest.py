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
    rngs = nnx.Rngs(42)
    key = rngs()

    # Common arrays and variables for examples
    x = jnp.array(
        [[1.0, 2.0], [3.0, 4.0]]
    )  # Common input array (batch, spatial, channels)
    log_density = jnp.zeros(2)  # Common log density
    eps = jax.random.normal(key, (32, 32))  # Random field for examples
    real_data = jax.random.normal(key, (32, 32))  # Real data for Fourier examples
    t = jnp.array(0.5)
    time_values = t  # Alias
    position_indices = jnp.arange(10)  # Position indices
    input_features = jax.random.normal(key, (5, 32))  # Input features
    features = jax.random.normal(key, (5, 64))  # Features array
    lattice_data = jax.random.normal(key, (8, 8, 1))  # Lattice data
    local_coupling = jnp.ones((16,))  # Local coupling for features
    lat = bijx.lie.sample_haar(key, 2, (4, 4, 2))  # Gauge field config
    # for symmetry; one matrix per vertex
    gauge_matrices = bijx.lie.sample_haar(key, 2, (4, 4))
    phi = jax.random.normal(key, (8, 8))  # Scalar field

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

    class SomeExpensiveDistribution(SomeArrayDistribution):
        def __init__(self):
            super().__init__(event_shape=(10,))

    class SomeVectorField(nnx.Module):
        def __call__(self, t, x, **kwargs):
            return -x, jnp.ones(x.shape[:-1])

    class SomeGaugeVF(nnx.Module):
        def __call__(self, t, u, **kwargs):
            return jnp.zeros_like(u), jnp.zeros(u.shape[:-2])

    # Create a simple model for noise_model example
    class SimpleModel(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(10, 1, rngs=rngs)

    class SomeBijection(bijx.Bijection):
        def __init__(self, rngs=None):
            self.a = nnx.Param(jnp.array(1.0))

        def forward(self, x, log_density, **kwargs):
            return x + self.a, log_density

        def reverse(self, x, log_density, **kwargs):
            return x - self.a, log_density

    model = SimpleModel(rngs)

    # Mock functions for examples
    def potential(u):
        return jnp.real(jnp.trace(u @ u.conj().T))

    def skew_symmetric(omega):
        """Create skew-symmetric matrix from 3D vector."""
        return jnp.array(
            [
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0],
            ]
        )

    # Add everything to doctest namespace
    doctest_namespace.update(
        {
            # Core imports
            "bijx": bijx,
            "jax": jax,
            "jnp": jnp,
            "nnx": nnx,
            # Common objects
            "rngs": nnx.Rngs(42),
            "key": key,
            "rng": key,
            "model": model,
            # Common arrays and variables
            "x": x,
            "log_density": log_density,
            "eps": eps,
            "real_data": real_data,
            "t": t,
            "time_values": time_values,
            "position_indices": position_indices,
            "input_features": input_features,
            "features": features,
            "lattice_data": lattice_data,
            "local_coupling": local_coupling,
            "lat": lat,
            "gauge_matrices": gauge_matrices,
            "phi": phi,
            "U": jnp.eye(2, dtype=complex),
            # Mock classes and functions
            "SomeBijection": SomeBijection,
            "SomeArrayDistribution": SomeArrayDistribution,
            "SomeExpensiveDistribution": SomeExpensiveDistribution,
            "SomeVectorField": SomeVectorField,
            "SomeGaugeVF": SomeGaugeVF,
            "potential": potential,
            "skew_symmetric": skew_symmetric,
            "SU2_GEN": bijx.lie.SU2_GEN,
        }
    )
