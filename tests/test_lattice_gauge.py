"""
Lattice gauge tests split from test_physics for focus and clarity.
"""

import jax
import jax.numpy as jnp
import numpy as np

from bijx.lattice import roll_lattice
from bijx.lattice.gauge import (
    apply_gauge_sym,
    flip_axis,
    rotate_lat,
    swap_axes,
    wilson_action,
)
from bijx.lie import sample_haar

from .utils import ATOL_RELAXED


class TestLatticeGaugeSymmetries:
    def _random_su2(self, key, shape):
        return sample_haar(key, 2, shape)

    def test_coordinate_symmetries_preserve_action(self, rng_key):
        key = rng_key
        lat_shape = (4, 4)
        u = self._random_su2(key, lat_shape + (2,))
        beta = 0.5
        s0 = wilson_action(u, beta)
        s_swap = wilson_action(swap_axes(u, 0, 1), beta)
        s_flip = wilson_action(flip_axis(u, 0), beta)
        s_rot = wilson_action(rotate_lat(u, 0, 1), beta)
        np.testing.assert_allclose(s0, s_swap, atol=ATOL_RELAXED)
        np.testing.assert_allclose(s0, s_flip, atol=ATOL_RELAXED)
        np.testing.assert_allclose(s0, s_rot, atol=ATOL_RELAXED)

    def test_gauge_transformation_preserves_action(self, rng_key):
        key = rng_key
        lat_shape = (3, 3)
        u = self._random_su2(key, lat_shape + (2,))
        gs = self._random_su2(jax.random.split(key)[0], lat_shape)
        beta = 1.0
        s0 = wilson_action(u, beta)
        s_g = wilson_action(apply_gauge_sym(u, gs), beta)
        np.testing.assert_allclose(s0, s_g, atol=ATOL_RELAXED)

    def test_roll_lattice_consistency(self):
        arr = jnp.arange(4 * 5).reshape(4, 5)
        rolled = roll_lattice(arr, (1, 0))
        assert rolled[0, 0] == arr[1, 0]
