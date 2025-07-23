"""
Spline-based bijections.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_autovmap import auto_vmap

from ..utils import ShapeInfo
from .base import BijectionApplyFn


@auto_vmap(inputs=0, bin_widths=1, bin_heights=1, knot_slopes=1)
def rational_quadratic_spline(
    inputs,
    bin_widths,
    bin_heights,
    knot_slopes,
    *,
    inverse=False,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_slope=1e-3,
):
    """Apply monotonic rational quadratic spline transformation.

    Following arXiv:1906.04032.
    Assumes inputs.shape = (..., n), parameters are (..., n, num_bins).
    """
    num_bins = bin_widths.shape[-1]

    # Normalize widths and heights using softmax
    bin_widths = nnx.softmax(bin_widths, axis=-1)
    bin_heights = nnx.softmax(bin_heights, axis=-1)

    # Enforce minimum bin size
    bin_widths = min_bin_width + (1 - min_bin_width * num_bins) * bin_widths
    bin_heights = min_bin_height + (1 - min_bin_height * num_bins) * bin_heights

    # Compute knot positions (cumulative sum gives knot positions)
    knot_x = jnp.pad(
        jnp.cumsum(bin_widths, -1),
        [(0, 0)] * (bin_widths.ndim - 1) + [(1, 0)],
        constant_values=0,
    )
    knot_y = jnp.pad(
        jnp.cumsum(bin_heights, -1),
        [(0, 0)] * (bin_heights.ndim - 1) + [(1, 0)],
        constant_values=0,
    )

    # Ensure positive slopes using softplus for internal knots
    softplus_scale = np.log(2) / (1 - min_slope)
    internal_slopes = (
        min_slope + nnx.softplus(softplus_scale * knot_slopes) / softplus_scale
    )

    # Pad with 1s for boundary slopes to match linear tails
    padding = [(0, 0)] * (internal_slopes.ndim - 1) + [(1, 1)]
    slopes = jnp.pad(internal_slopes, padding, constant_values=1.0)

    # Handle inputs outside the [0, 1] spline domain,
    # where the transform is the identity.
    in_bounds = (inputs >= 0) & (inputs <= 1)
    # Clamp inputs for internal calculations to avoid NaNs.
    inputs_clipped = jnp.clip(inputs, 0, 1)

    # Helper function for advanced indexing over the batch dimension

    # Find which bin each input falls into, using the clipped inputs.
    if inverse:
        bin_idx = jnp.searchsorted(knot_y, inputs_clipped, side="right") - 1
    else:
        bin_idx = jnp.searchsorted(knot_x, inputs_clipped, side="right") - 1

    # Get bin boundaries and slopes for each input
    left_knot_x = knot_x[bin_idx]
    bin_width = bin_widths[bin_idx]
    left_knot_y = knot_y[bin_idx]
    left_slope = slopes[bin_idx]
    bin_height = bin_heights[bin_idx]
    right_slope = slopes[bin_idx + 1]

    # Compute bin slope (average rise over run)
    bin_slope = bin_height / bin_width

    if inverse:
        # Solve quadratic equation for inverse transform
        y_offset = inputs_clipped - left_knot_y

        quad_a = y_offset * (left_slope + right_slope - 2 * bin_slope) + bin_height * (
            bin_slope - left_slope
        )
        quad_b = bin_height * left_slope - y_offset * (
            left_slope + right_slope - 2 * bin_slope
        )
        quad_c = -bin_slope * y_offset

        discriminant = quad_b**2 - 4 * quad_a * quad_c
        normalized_pos = (2 * quad_c) / (-quad_b - jnp.sqrt(discriminant))
        outputs_spline = normalized_pos * bin_width + left_knot_x

        # Compute log determinant for the forward transform dy/dx
        pos_complement = normalized_pos * (1 - normalized_pos)
        denominator = bin_slope + (
            (left_slope + right_slope - 2 * bin_slope) * pos_complement
        )
        numerator = bin_slope**2 * (
            right_slope * normalized_pos**2
            + 2 * bin_slope * pos_complement
            + left_slope * (1 - normalized_pos) ** 2
        )
        log_det_spline = jnp.log(numerator) - 2 * jnp.log(denominator)

        # For the inverse transform, we need -log_det(dy/dx)
        log_det = -log_det_spline

    else:  # Forward transform
        normalized_pos = (inputs_clipped - left_knot_x) / bin_width

        numerator_term = bin_slope * normalized_pos**2 + left_slope * normalized_pos * (
            1 - normalized_pos
        )
        denominator_term = bin_slope + (
            right_slope + left_slope - 2 * bin_slope
        ) * normalized_pos * (1 - normalized_pos)
        outputs_spline = left_knot_y + bin_height * numerator_term / denominator_term

        derivative_numerator = bin_slope**2 * (
            right_slope * normalized_pos**2
            + 2 * bin_slope * normalized_pos * (1 - normalized_pos)
            + left_slope * (1 - normalized_pos) ** 2
        )
        log_det = jnp.log(derivative_numerator) - 2 * jnp.log(denominator_term)

    # For out-of-bounds inputs, the transform is the identity.
    # The output is the input, and the log_det is 0.
    outputs = jnp.where(in_bounds, outputs_spline, inputs)
    final_log_det = jnp.where(in_bounds, log_det, 0.0)

    return outputs, final_log_det


class MonotoneRQSpline(BijectionApplyFn):
    def __init__(
        self,
        knots,
        event_shape=(),
        *,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_slope=1e-3,
        widths_init=nnx.initializers.normal(),
        heights_init=nnx.initializers.normal(),
        slopes_init=nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        """
        Monotone rational quadratic spline.

        Assume input is 1-dimensional.
        """
        self.event_shape = event_shape
        self.in_features = np.prod(event_shape, dtype=int)

        widths = widths_init(rngs.params(), (*event_shape, knots))
        heights = heights_init(rngs.params(), (*event_shape, knots - 1))
        slopes = slopes_init(rngs.params(), (*event_shape, knots - 1))

        self.widths = nnx.Param(widths)
        self.heights = nnx.Param(heights)
        self.slopes = nnx.Param(slopes)

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_slope = min_slope
        self.knots = knots

    @property
    def param_count(self):
        """Total number of parameters needed: widths + heights + slopes."""
        return 3 * self.knots - 1

    @property
    def param_splits(self):
        """How to split the parameter vector into widths, heights, and slopes."""
        return [self.knots, self.knots, self.knots - 1]

    def apply(self, x, log_density, reverse, **kwargs):
        # flatten event_shape of x (last len(event_shape) dimensions) using einops
        # if len(self.event_shape) == 0:
        #     x = jnp.expand_dims(x, -1)
        # else:
        #     axes = ' '.join(f'a{i}' for i in range(len(self.event_shape)))
        #     axes_sizes = {
        #         f'a{i}': s
        #         for i, s in enumerate(self.event_shape)
        #     }
        #     x = rearrange(x, f'... {axes} -> ... ({axes})', **axes_sizes)

        x, log_jac = rational_quadratic_spline(
            x,
            self.widths.value,
            self.heights.value,
            self.slopes.value,
            inverse=reverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_slope=self.min_slope,
        )

        # return to original event shape
        # if len(self.event_shape) == 0:
        #     x = jnp.squeeze(x, -1)
        # else:
        #     x = rearrange(x, f'... ({axes}) -> ... {axes}', **axes_sizes)

        event_axes = ShapeInfo(event_shape=self.event_shape).event_axes
        return x, log_density - jnp.sum(log_jac, axis=event_axes)
