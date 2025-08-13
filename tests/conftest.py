"""
Shared test configuration and fixtures for the bijx test suite.

Goals:
- Default to 64-bit precision.
- Reusable deterministic RNG by test node id.
"""

import hashlib
import os

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

# Set default precision to 64-bit
jax.config.update("jax_enable_x64", True)

# Numpy printing for debugging
np.set_printoptions(suppress=True, linewidth=120)

# Global Hypothesis configuration
settings.register_profile(
    "dev",
    deadline=None,  # avoid flaky timeouts with JAX/JIT/XLA
    max_examples=int(os.getenv("HYP_MAX_EXAMPLES", "10")),
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# Deterministic seed per test node id
# Note: CANNOT use this in combination with hypothesis.
# In the latter case, use integer random seeds instead.


def _seed_from_nodeid(nodeid: str) -> int:
    h = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


@pytest.fixture
def rng_key(request: pytest.FixtureRequest) -> jax.Array:
    seed = _seed_from_nodeid(request.node.nodeid)
    return jax.random.key(seed)
